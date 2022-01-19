"""Psycho-acoustic model for lossy audio encoder (inspired by aac encoder)
Loosely based on code from Gerald Schuller, June 2018
(https://github.com/TUIlmenauAMS/Python-Audio-Coder) See also chapter 9 in
"Digital Audio Signal Processing" by Udo Zolzer

"""

import torch


class PsychoacousticModel():
    """A psycho-acoustic model"""
    def __init__(self, sample_rate, device, filter_bands_n=1024,
                 bark_bands_n=64, alpha=0.6, compute_dtype=torch.float32):

        """Computes required initialization matrices.

        For standard MP3 encoding, they use filter_bands_n=1024 and
        bark_bands_n=64.  If one deviates from these parameters, both the
        levels of the global masking threshold and the quiet threshold can
        vary, since both are established in the bark scale and then converted
        to the frequency scale.  In that conversion process, the bark energy
        gets dissipated into (more) frequency buckets and the frequency by
        frequency threshold is hence lowered then more frequency buckets or the
        less bark bands one has.  todo: add normalization coefficient in
        definition of W and W_inv to solve this issue


        :param sample_rate:       sample_rate
        :param alpha:             exponent for non-linear superposition
                                  lower means lower quality (1.0 is linear
                                  superposition, 0.6 is default)
        :param filter_bands_n:    number of filter bands of the filter bank
        :param bark_bands_n:      number of bark bands

        :param compute_dtype      The compute dtype of model. Inputs dtype must
                                  match compute_dtype (no implicit casting) Can
                                  be tf.float64, tf.float32 or tf.bfloat16
                                  (defaults to tf.float32)

                                  Note: tf.float16 is not allowed since float16
                                  does not allow for enough range in the
                                  exponent of the amplitudes to get correct
                                  results :return: tuple with pre-computed
                                  required for encoder and decoder :raises
                                  TypeError when compute_dtype is not

        """

        self.alpha = alpha
        self.sample_rate = sample_rate
        self.bark_bands_n = bark_bands_n
        self.filter_bands_n = filter_bands_n
        self.device = device

        # Compute data type
        self.compute_dtype = compute_dtype

        # Definition of dB range
        self._dB_MAX = torch.tensor(120, dtype=self.compute_dtype,
                                    device=self.device)

        self._INTENSITY_EPS = torch.tensor(1e-14, dtype=self.compute_dtype,
                                           device=self.device)
        self._dB_MIN = self.amplitude_to_dB(self._INTENSITY_EPS)

        # pre-compute some values & matrices with higher precision, then
        # down-cast to compute_dtype
        precompute_dtype = torch.float64
        # Nyquist frequency: maximum frequency given a sample rate
        self.max_frequency = torch.tensor(
            self.sample_rate,
            dtype=precompute_dtype,
            device=self.device) / 2.0

        self.max_bark = self.freq2bark(self.max_frequency)
        self.bark_band_width = self.max_bark / self.bark_bands_n

        W, W_inv = self._bark_freq_mapping(precompute_dtype=precompute_dtype)
        self.W = W.type(compute_dtype)
        self.W_inv = W_inv.type(compute_dtype)
        self.quiet_threshold_intensity = \
            self._quiet_threshold_intensity_in_bark(
                precompute_dtype=precompute_dtype).type(compute_dtype)
        self.spreading_matrix = self._spreading_matrix_in_bark().type(
            compute_dtype).to(self.device)

    def amplitude_to_dB(self, mdct_amplitude):
        """Convert amplitude in [-1,1] to dB scale such that :
        1. an amplitude squared (intensity) of 1 corresponds to the maximum dB
           level (_dB_MAX), and
        2. an amplitude squared (intensity) of _INTENSITY_EPS corresponds with
           the minimum dB level (_dB_MIN)

        Args:
            mdct_amplitude (tensor): normalized amplitudes to convert to dB
            scale

        Returns:
            tensor: amplitudes in dB scale in[_dB_MIN,_dB_MaX] dtype = compute
            dtype

        """
        ampl_dB = (10*torch.log10(
            torch.max(self._INTENSITY_EPS, mdct_amplitude ** 2))
                   + self._dB_MAX).type(self.compute_dtype)
        return ampl_dB

    def amplitude_to_dB_norm(self, mdct_amplitude):
        """Utility fonction to convert normalized amplitude in [-1,1] to the normalized
        dB scale in [0,1]

        Args:
            mdct_amplitude (tensor): amplitude normalized in [-1,1]

        Returns :
            tensor : amplitudes in normalized dB scale in [0,1]
            dtype = compute dtype

        """
        ampl_dB = self.amplitude_to_dB(mdct_amplitude)
        return (ampl_dB - self._dB_MIN) / (self._dB_MAX - self._dB_MIN)

    def tonality(self, mdct_amplitudes):
        """compute tonality from the spectral flateness measure (SFM)

        Args:
            mdct_amplitudes (tensor): mdct amplitudes (spectrum) for each
            filter. Shape : [batches_n, blocks_n, filter_bands_n, channels_n]


        Returns:
            tensor: tonality vector. Shape : [batches_n,blocks_n,1,channels_n]
        """
        mdct_intensity = torch.pow(mdct_amplitudes, 2)
        geometric_mean = torch.exp(torch.mean(torch.log(torch.max(
            self._INTENSITY_EPS, mdct_intensity)), axis=2, keepdim=True))
        arithmetic_mean = torch.mean(mdct_intensity, axis=2, keepdim=True)
        sfm = 10. * torch.log(torch.divide(geometric_mean,
                                           arithmetic_mean +
                                           self._INTENSITY_EPS)) \
            / torch.log(torch.tensor(10))

        sfm = torch.minimum(sfm / -60., torch.tensor(1.0))

        return sfm

    def global_masking_threshold(
            self,
            mdct_amplitudes,
            tonality_per_block,
            drown=0):
        """Determines which amplitudes we cannot hear, either since they are too soft
        to hear or because other louder amplitudes are masking it.
        Method uses non-linear superposition determined by factor self.alpha

        Args:
            mdct_amplitudes (tensor): mdct amplitudes (spectrum).
              Shape : [batches_n, blocks_n, filter_bands_n, channels_n]
            tonality_per_block (tensor): tonality vector associated with
              mdct_amplitudes.
              Shape : [batches_n, blocks_n, 1, channels_n]
            drown (float, optional): factor 0..1 to drown out audible sounds
              (0: no drowning, 1: fully drowned). defaults to 0

        Returns:
            tensor: masking threshold in amplitude. Never negative.
              Shape : [batches_n, blocks_n, filter_bands_n, channels_n]
        """

        # with tf.name_scope('global_masking_threshold'):
        masking_intensity = self._masking_intensity_in_bark(
            mdct_amplitudes, tonality_per_block, drown)

        # Take max between quiet threshold and masking threshold
        # Note: even though both thresholds are expressed as amplitudes,
        # they are all positive due to the way they were constructed
        global_masking_intensity_in_bark = torch.maximum(
            masking_intensity, self.quiet_threshold_intensity)

        global_mask_threshold = self._bark_intensity_to_freq_ampl(
            global_masking_intensity_in_bark)

        return global_mask_threshold

    def add_noise(self, mdct_amplitudes, masking_threshold):
        """Adds inaudible noise to amplitudes, using the masking threshold.

        The noise added is calibrated at a 3-sigma deviation in both
        directions: masking_threshold = 6*sigma As such, there is a 0.2%
        probability that the noise added is bigger than the masking_threshold

        Args:
            mdct_amplitudes (tensor): mdct amplitudes for each filter.
              Shape : [batches_n, blocks_n, filter_bands_n, channels_n]
            masking_threshold (tensor)): masking threshold in amplitude.
              Always positive.
              Shape : [batches_n, blocks_n, filter_bands_n, channels_n]

        Returns:
            tensor : mdct amplitudes with inaudible noise added

        """
        noise = masking_threshold * torch.normal(
            torch.zeros_like(masking_threshold),
            torch.ones_like(masking_threshold)/6)
        return mdct_amplitudes + noise

    def _masking_intensity_in_bark(
            self,
            mdct_amplitudes,
            tonality_per_block,
            drown=0.0):
        """Returns amplitudes that are masked by the sound defined by mdct_amplitudes

        Args:
            mdct_amplitudes (tensor): mdct amplitudes (spectrum) for each
              filter.
              Shape : [batches_n, blocks_n, filter_bands_n, channels_n]
              Should be of compute_dtype
            tonality_per_block (tensor): tonality vector associated with the
              mdct_amplitudes.
              Shape : [batches_n, blocks_n, 1, channels_n]
              Should be of compute_dtype
            drown (float, optional): factor 0..1 to drown out audible sounds
              (0: no drowning, 1: fully drowned). defaults to 0.

        Returns:
            tensor: vector of intensities for softest audible sounds given a
              certain sound
              Shape : [batches_n, blocks_n, bark_bands_n, channels_n]
        """
        # compute masking offset:
        #    O(i) = tonality (14.5 + i) + (1 - tonality) 5.5
        # with i the bark index
        # see p10 ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)
        # in einsum, we tf.squeeze() axis=2 (index i) and take outer product
        # with tf.linspace()
        offset = (1. - drown) * (
            torch.einsum('nbic,j->nbjc',
                         tonality_per_block,
                         torch.linspace(
                             torch.tensor(
                                 0.0, dtype=self.compute_dtype,
                                 device=self.device
                             ),
                             self.max_bark,
                             self.bark_bands_n, device=self.device)
                         )
            + 9. * tonality_per_block
            + 5.5)
        # add offset to spreading matrix
        # (see (9.18) in "Digital Audio Signal Processing" by Udo Zolzer)
        # note: einsum('.j.,.j.->.j.') multiplies elements on diagonal
        # element-wise (without summing over j)
        masking_matrix = torch.einsum(
            'ij,nbjc->nbijc',
            self.spreading_matrix,
            torch.pow(10, -self.alpha * offset / 10.0))

        # Transposed version of (9.17) in Digital Audio Signal Processing by
        # Udo Zolzer
        # \Sum_i (amplitude_i^2)^{\alpha} x [ mask^{\alpha}_{i-n} ]_n
        #   = \Sum amplitude_i x mask_{in}              --> each row is a mask

        # Non-linear superposition (see p13
        # ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)

        # \alpha ~ 0.3 is valid for 94 bark_bands_n; with less bark_bands_n 0.3
        # leads to (way) too much masking
        intensities_in_bark = self._to_bark_intensity(mdct_amplitudes)
        masking_intensity_in_bark = torch.einsum(
            'nbic,nbijc->nbjc',
            torch.pow(torch.maximum(self._INTENSITY_EPS, intensities_in_bark),
                      self.alpha),
            masking_matrix)
        masking_intensity_in_bark = torch.pow(
            torch.maximum(self._INTENSITY_EPS, masking_intensity_in_bark),
            1. / self.alpha)

        return masking_intensity_in_bark

    def _spreading_matrix_in_bark(self):
        """Returns (power) spreading matrix, to apply in bark scale to determine
        masking threshold from sound

        Returns:
            tensor : spreading matrix [bark_bands_n, bark_bands_n]

        """

        # Prototype spreading function [Bark/dB] see equation (9.15) in
        # "Digital Audio Signal Processing" by Udo Zolzer
        z = torch.linspace(-self.max_bark,
                           self.max_bark, 2 * self.bark_bands_n)
        f_spreading = 15.81 + 7.5 * (z + 0.474)\
            - 17.5 * torch.sqrt(1 + torch.pow(z + 0.474, 2))

        # Convert from dB to intensity and include alpha exponent
        f_spreading_intensity = torch.pow(10.0,
                                          self.alpha * f_spreading / 10.0)

        # Turns the spreading prototype function into a (bark_bands_n x
        # bark_bands_n) matrix of shifted versions.  Transposed version of
        # (9.17) in Digital Audio Signal Processing by Udo Zolzer

        spreading_matrix = torch.stack(
            [f_spreading_intensity[(self.bark_bands_n - row):
                                   (2 * self.bark_bands_n - row)]
             for row in range(self.bark_bands_n)], axis=0)

        return spreading_matrix

    def _quiet_threshold_intensity_in_bark(self, precompute_dtype):
        """Compute the intensity of the softest sounds one can hear
           See (9.3) in "Digital Audio Signal Processing" by Udo Zolzer

        Returns: tensor : intensity vector for softest audible sounds [1, 1,
            bark_bands_n, 1] returned amplitudes are all positive

        """
        # Threshold in quiet:
        bark_bands_mid_bark = self.bark_band_width\
            * torch.arange(self.bark_bands_n,
                           dtype=precompute_dtype,
                           device=self.device)\
            + self.bark_band_width / 2.

        bark_bands_mid_kHz = self.bark2freq(bark_bands_mid_bark) / 1000.
        # Threshold of quiet expressed as amplitude in dB-scale for each Bark
        # bands. see (9.3) in "Digital Audio Signal Processing" by Udo Zolzer
        quiet_threshold_dB = torch.clip(
            (3.64 * (torch.pow(bark_bands_mid_kHz, -0.8))
             - 6.5 * torch.exp(-0.6 * torch.pow(bark_bands_mid_kHz - 3.3, 2.))
             + 1e-3 * (torch.pow(bark_bands_mid_kHz, 4.))),
            self._dB_MIN.type(precompute_dtype),
            self._dB_MAX.type(precompute_dtype))
        # convert to amplitude scale, where _dB_MAX corresponds with an
        # amplitude of 1.0
        quiet_threshold_intensity = torch.pow(
            torch.tensor(10.0, dtype=precompute_dtype),
            (quiet_threshold_dB - self._dB_MAX.type(precompute_dtype)) / 10)
        return quiet_threshold_intensity.view(1, 1, -1, 1)

    def _bark_freq_mapping(self, precompute_dtype):
        """Compute (static) mapping between MDCT filter bank ranges and Bark
        bands.
        they fall in

                                  ----> bark_bands_n
                       |        [ 1  0  ...  0 ]
          W =          |        [ 0  1  ...  0 ]
                       V        [ 0  1  ...  0 ]
                  filter_bank_n [       ...    ]
                                [ 0  0  ...  1 ]

        Inverse transformation, from Bark bins amplitude(!) to MDCT filter
        bands amplitude(!), consists of transposed with normalization factor
        such that power (square of amplitude) in each Bark band gets split
        equally between the filter bands making up that Bark band:


                          ----> filter_bank_n
               |        [ 1/\\sqrt{1} 0           0          ...  0          ]
       W_inv = |        [ 0          1/\\sqrt{2}  1/\\sqrt{2} ...  0         ]
               V        [                                   ...              ]
          bark_bands_n  [ 0          0           0          ...  1/\\sqrt{1} ]

        Returns:
            2 tensors with shape
                      W      [filter_bank_n , bark_band_n]
                      W_inv  [bark_band_n   , filter_bank_n]

        """
        filter_band_width = self.max_frequency / self.filter_bands_n

        def freq_interval_overlap(freq_index, bark_index):
            bark_low = self.bark_band_width * bark_index
            bark_low_in_Hz = torch.broadcast_to(
                self.bark2freq(bark_low),
                [self.filter_bands_n, self.bark_bands_n])
            bark_high_in_Hz = torch.broadcast_to(
                self.bark2freq(bark_low + self. bark_band_width),
                [self.filter_bands_n, self.bark_bands_n])

            freq_low = filter_band_width * freq_index
            bark_low_in_Hz_clipped = torch.clip(
                bark_low_in_Hz,
                freq_low,
                freq_low + filter_band_width)
            bark_high_in_Hz_clipped = torch.clip(
                bark_high_in_Hz,
                freq_low,
                freq_low + filter_band_width)

            overlap = bark_high_in_Hz_clipped - bark_low_in_Hz_clipped
            return (
                overlap / filter_band_width,
                overlap / (bark_high_in_Hz - bark_low_in_Hz))

        bark_columns = torch.arange(
            self.bark_bands_n,
            dtype=precompute_dtype,
            device=self.device
        ).view(1, -1)

        freq_rows = torch.arange(
            self.filter_bands_n,
            dtype=precompute_dtype,
            device=self.device
        ).view(-1, 1)

        W, W_inv_transpose = freq_interval_overlap(freq_rows, bark_columns)

        return W, torch.transpose(W_inv_transpose, 0, 1)

    def _to_bark_intensity(self, mdct_amplitudes):
        """Takes MDCT amplitudes and maps it into Bark bands amplitudes.

        Power spectral density of Bark band is sum of power spectral density in
        corresponding filter bands (power spectral density of signal S = X_1^2
        + ... + X_n^2)

          (see also slide p9 of
          ./docs/05_shl_AC_Psychacoustics_Models_WS-2016-17_gs.pdf)

        Args :
            param mdct_amplitudes(tensor) :  vector of mdct amplitudes
              (spectrum) for each filter
              [batches_n, blocks_n, filter_bands_n, channels_n]
            param W(tensor) : matrix to convert from filter bins to bark bins
              [filter_bands_n, bark_bands_n]

        Returns :
            tensor : vector of intensities per Bark band
              [batches_n, blocks_n, bark_bands_n, channels_n]

        """
        # tf.maximum() is necessary, to make sure rounding errors don't make
        # the gradient nan!
        mdct_intensity = torch.pow(mdct_amplitudes, 2)
        mdct_intensity_in_bark = torch.einsum(
            'nbic,ij->nbjc', mdct_intensity, self.W)

        return mdct_intensity_in_bark

    def _bark_intensity_to_freq_ampl(self, bark_intensity):
        """Takes Bark band intensity and maps it to MDCT amplitudes

        Power spectral density of Bark band is split equally between the filter
        bands making up the Bark band (one-to-many). As a result, intensities
        in the bark scale get smeared out in the frequency scale.  For higher
        frequencies, this smearing effect becomes more important.  As a result,
        e.g. the quiet threshold which is specified in the bark scale, gets
        smeared out in the frequency scale (and looks hence lower in value!)

        Args :
            param bark_intensity(tensor) : vector of signal intensities in
              the Bark bands [batches_n, blocks_n, bark_bands_n, channels_n]
            param W_inv(tensor) : matrix to convert from filter bins
              to bark bins [bark_bands_n, filter_bands_n]

        Returns:
            (tensor) : vector of mdct amplitudes (spectrum) for each filter
              [batches_n, blocks_n, filter_bands_n, channels_n]

        """
        mdct_intensity = torch.einsum(
            'nbic,ij->nbjc', bark_intensity, self.W_inv)
        return torch.pow(torch.maximum(
            self._INTENSITY_EPS, mdct_intensity), 0.5)

    def freq2bark(self, frequencies):
        """Empirical Bark scale"""
        return 6. * torch.asinh(frequencies / 600.)

    def bark2freq(self, bark_band):
        """Empirical Bark scale"""
        return 600. * torch.sinh(bark_band / 6.)

    def apply_psycho_single(self, mdct_amplitudes):
        """Computes all the necessary indicators and add noise to a single mdct

        Args:
            mdct_amplitudes (tensor): mdct amplitudes
              [channels_n, blocks_n, filter_bands_n]
        Returns:
            tensor : mdct amplitudes with added noise
        """
        mdct_amplitudes = mdct_amplitudes[None, :]
        # Reshaping to needed shapes
        mdct_amplitudes = mdct_amplitudes.transpose(1, 2)
        mdct_amplitudes = mdct_amplitudes.transpose(2, 3)

        tonality = self.tonality(mdct_amplitudes)
        global_masking = self.global_masking_threshold(mdct_amplitudes,
                                                       tonality)
        mdct_amp_noise = self.add_noise(mdct_amplitudes, global_masking)

        # Reshaping to input shape
        mdct_amp_noise = mdct_amp_noise.transpose(2, 3)
        mdct_amp_noise = mdct_amp_noise.transpose(1, 2)

        mdct_amp_noise = mdct_amp_noise[0]

        return mdct_amp_noise

    def apply_psycho_batch(self, mdct_amplitudes):
        """Computes all the necessary indicators and add noise to the
        mdct in batch

        Args:
            mdct_amplitudes (tensor): mdct amplitudes
              [batches_n, channels_n, blocks_n, filter_bands_n]

        Returns:
            tensor : mdct amplitudes with added noise
        """
        # mdct_amplitudes = mdct_amplitudes[None, :]
        # Reshaping to needed shapes
        mdct_amplitudes = mdct_amplitudes.transpose(1, 2)
        mdct_amplitudes = mdct_amplitudes.transpose(2, 3)

        tonality = self.tonality(mdct_amplitudes)
        global_masking = self.global_masking_threshold(mdct_amplitudes,
                                                       tonality)
        mdct_amp_noise = self.add_noise(mdct_amplitudes,
                                        global_masking)
        # Reshaping to input shape
        mdct_amp_noise = mdct_amp_noise.transpose(2, 3)
        mdct_amp_noise = mdct_amp_noise.transpose(1, 2)

        return mdct_amp_noise
