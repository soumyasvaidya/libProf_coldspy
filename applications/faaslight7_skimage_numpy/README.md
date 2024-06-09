- Application Name: Skimage_numpy
- App Repo: https://github.com/ryfeus/lambda-packs/tree/master/Skimage_numpy
- Skimage Repo: https://github.com/scikit-image/scikit-image
- Version: 0.22.0

Optimization: Removed scipy because it is not needed

### Average Initialization latency of
- Original code: 1569.2704436860067
- Optimized code: 1114.3507941653158

#### Average Intialization latency reduced: 28.98%

### Average End to End latency of
- Original code: 1821.7277303754265
- Optimized code: 1342.4794489465155

#### Average End to End latency reduced: 26.30%

### Average Memory Utilization of
- Original code: 112.09215017064847
- Optimized code: 112.20583468395462

#### Average Memory Utilization increased: 0.10%


## 100+ Cold Starts
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)

Optimization diff
```diff
diff -x *.pyc -bur --co original/package/skimage/color/colorconv.py optimized/package/skimage/color/colorconv.py
--- original/package/skimage/color/colorconv.py	2024-04-26 19:16:30
+++ optimized/package/skimage/color/colorconv.py	2024-05-08 16:47:08
@@ -52,7 +52,7 @@
 from warnings import warn
 
 import numpy as np
-from scipy import linalg
+# from scipy import linalg
 
 
 from .._shared.utils import (
@@ -74,72 +74,72 @@
     from numpy.exceptions import AxisError
 
 
-def convert_colorspace(arr, fromspace, tospace, *, channel_axis=-1):
-    """Convert an image array to a new color space.
+# def convert_colorspace(arr, fromspace, tospace, *, channel_axis=-1):
+#     """Convert an image array to a new color space.
 
-    Valid color spaces are:
-        'RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'
+#     Valid color spaces are:
+#         'RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'
 
-    Parameters
-    ----------
-    arr : (..., 3, ...) array_like
-        The image to convert. By default, the final dimension denotes
-        channels.
-    fromspace : str
-        The color space to convert from. Can be specified in lower case.
-    tospace : str
-        The color space to convert to. Can be specified in lower case.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     arr : (..., 3, ...) array_like
+#         The image to convert. By default, the final dimension denotes
+#         channels.
+#     fromspace : str
+#         The color space to convert from. Can be specified in lower case.
+#     tospace : str
+#         The color space to convert to. Can be specified in lower case.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The converted image. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The converted image. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If fromspace is not a valid color space
-    ValueError
-        If tospace is not a valid color space
+#     Raises
+#     ------
+#     ValueError
+#         If fromspace is not a valid color space
+#     ValueError
+#         If tospace is not a valid color space
 
-    Notes
-    -----
-    Conversion is performed through the "central" RGB color space,
-    i.e. conversion from XYZ to HSV is implemented as ``XYZ -> RGB -> HSV``
-    instead of directly.
+#     Notes
+#     -----
+#     Conversion is performed through the "central" RGB color space,
+#     i.e. conversion from XYZ to HSV is implemented as ``XYZ -> RGB -> HSV``
+#     instead of directly.
 
-    Examples
-    --------
-    >>> from skimage import data
-    >>> img = data.astronaut()
-    >>> img_hsv = convert_colorspace(img, 'RGB', 'HSV')
-    """
-    fromdict = {'rgb': identity, 'hsv': hsv2rgb, 'rgb cie': rgbcie2rgb,
-                'xyz': xyz2rgb, 'yuv': yuv2rgb, 'yiq': yiq2rgb,
-                'ypbpr': ypbpr2rgb, 'ycbcr': ycbcr2rgb, 'ydbdr': ydbdr2rgb}
-    todict = {'rgb': identity, 'hsv': rgb2hsv, 'rgb cie': rgb2rgbcie,
-              'xyz': rgb2xyz, 'yuv': rgb2yuv, 'yiq': rgb2yiq,
-              'ypbpr': rgb2ypbpr, 'ycbcr': rgb2ycbcr, 'ydbdr': rgb2ydbdr}
+#     Examples
+#     --------
+#     >>> from skimage import data
+#     >>> img = data.astronaut()
+#     >>> img_hsv = convert_colorspace(img, 'RGB', 'HSV')
+#     """
+#     fromdict = {'rgb': identity, 'hsv': hsv2rgb, 'rgb cie': rgbcie2rgb,
+#                 'xyz': xyz2rgb, 'yuv': yuv2rgb, 'yiq': yiq2rgb,
+#                 'ypbpr': ypbpr2rgb, 'ycbcr': ycbcr2rgb, 'ydbdr': ydbdr2rgb}
+#     todict = {'rgb': identity, 'hsv': rgb2hsv, 'rgb cie': rgb2rgbcie,
+#               'xyz': rgb2xyz, 'yuv': rgb2yuv, 'yiq': rgb2yiq,
+#               'ypbpr': rgb2ypbpr, 'ycbcr': rgb2ycbcr, 'ydbdr': rgb2ydbdr}
 
-    fromspace = fromspace.lower()
-    tospace = tospace.lower()
-    if fromspace not in fromdict:
-        msg = f'`fromspace` has to be one of {fromdict.keys()}'
-        raise ValueError(msg)
-    if tospace not in todict:
-        msg = f'`tospace` has to be one of {todict.keys()}'
-        raise ValueError(msg)
+#     fromspace = fromspace.lower()
+#     tospace = tospace.lower()
+#     if fromspace not in fromdict:
+#         msg = f'`fromspace` has to be one of {fromdict.keys()}'
+#         raise ValueError(msg)
+#     if tospace not in todict:
+#         msg = f'`tospace` has to be one of {todict.keys()}'
+#         raise ValueError(msg)
 
-    return todict[tospace](
-        fromdict[fromspace](arr, channel_axis=channel_axis),
-        channel_axis=channel_axis
-    )
+#     return todict[tospace](
+#         fromdict[fromspace](arr, channel_axis=channel_axis),
+#         channel_axis=channel_axis
+#     )
 
 
 def _prepare_colorarray(arr, force_copy=False, *, channel_axis=-1):
@@ -407,7 +407,7 @@
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]])
 
-rgb_from_xyz = linalg.inv(xyz_from_rgb)
+# rgb_from_xyz = linalg.inv(xyz_from_rgb)
 
 # From https://en.wikipedia.org/wiki/CIE_1931_color_space
 # Note: Travis's code did not have the divide by 0.17697
@@ -415,11 +415,11 @@
                             [0.17697, 0.81240, 0.01063],
                             [0.00, 0.01, 0.99]]) / 0.17697
 
-rgbcie_from_xyz = linalg.inv(xyz_from_rgbcie)
+# rgbcie_from_xyz = linalg.inv(xyz_from_rgbcie)
 
 # construct matrices to and from rgb:
-rgbcie_from_rgb = rgbcie_from_xyz @ xyz_from_rgb
-rgb_from_rgbcie = rgb_from_xyz @ xyz_from_rgbcie
+# rgbcie_from_rgb = rgbcie_from_xyz @ xyz_from_rgb
+# rgb_from_rgbcie = rgb_from_xyz @ xyz_from_rgbcie
 
 
 gray_from_rgb = np.array([[0.2125, 0.7154, 0.0721],
@@ -430,31 +430,31 @@
                          [-0.14714119, -0.28886916,  0.43601035 ],
                          [ 0.61497538, -0.51496512, -0.10001026 ]])
 
-rgb_from_yuv = linalg.inv(yuv_from_rgb)
+# rgb_from_yuv = linalg.inv(yuv_from_rgb)
 
 yiq_from_rgb = np.array([[0.299     ,  0.587     ,  0.114     ],
                          [0.59590059, -0.27455667, -0.32134392],
                          [0.21153661, -0.52273617,  0.31119955]])
 
-rgb_from_yiq = linalg.inv(yiq_from_rgb)
+# rgb_from_yiq = linalg.inv(yiq_from_rgb)
 
 ypbpr_from_rgb = np.array([[ 0.299   , 0.587   , 0.114   ],
                            [-0.168736,-0.331264, 0.5     ],
                            [ 0.5     ,-0.418688,-0.081312]])
 
-rgb_from_ypbpr = linalg.inv(ypbpr_from_rgb)
+# rgb_from_ypbpr = linalg.inv(ypbpr_from_rgb)
 
 ycbcr_from_rgb = np.array([[    65.481,   128.553,    24.966],
                            [   -37.797,   -74.203,   112.0  ],
                            [   112.0  ,   -93.786,   -18.214]])
 
-rgb_from_ycbcr = linalg.inv(ycbcr_from_rgb)
+# rgb_from_ycbcr = linalg.inv(ycbcr_from_rgb)
 
 ydbdr_from_rgb = np.array([[    0.299,   0.587,    0.114],
                            [   -0.45 ,  -0.883,    1.333],
                            [   -1.333,   1.116,    0.217]])
 
-rgb_from_ydbdr = linalg.inv(ydbdr_from_rgb)
+# rgb_from_ydbdr = linalg.inv(ydbdr_from_rgb)
 
 
 # CIE LAB constants for Observer=2A, Illuminant=D65
@@ -625,7 +625,7 @@
 rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]])
-hed_from_rgb = linalg.inv(rgb_from_hed)
+# hed_from_rgb = linalg.inv(rgb_from_hed)
 
 # Following matrices are adapted form the Java code written by G.Landini.
 # The original code is available at:
@@ -636,68 +636,68 @@
                          [0.268, 0.570, 0.776],
                          [0.0, 0.0, 0.0]])
 rgb_from_hdx[2, :] = np.cross(rgb_from_hdx[0, :], rgb_from_hdx[1, :])
-hdx_from_rgb = linalg.inv(rgb_from_hdx)
+# hdx_from_rgb = linalg.inv(rgb_from_hdx)
 
 # Feulgen + Light Green
 rgb_from_fgx = np.array([[0.46420921, 0.83008335, 0.30827187],
                          [0.94705542, 0.25373821, 0.19650764],
                          [0.0, 0.0, 0.0]])
 rgb_from_fgx[2, :] = np.cross(rgb_from_fgx[0, :], rgb_from_fgx[1, :])
-fgx_from_rgb = linalg.inv(rgb_from_fgx)
+# fgx_from_rgb = linalg.inv(rgb_from_fgx)
 
 # Giemsa: Methyl Blue + Eosin
 rgb_from_bex = np.array([[0.834750233, 0.513556283, 0.196330403],
                          [0.092789, 0.954111, 0.283111],
                          [0.0, 0.0, 0.0]])
 rgb_from_bex[2, :] = np.cross(rgb_from_bex[0, :], rgb_from_bex[1, :])
-bex_from_rgb = linalg.inv(rgb_from_bex)
+# bex_from_rgb = linalg.inv(rgb_from_bex)
 
 # FastRed + FastBlue +  DAB
 rgb_from_rbd = np.array([[0.21393921, 0.85112669, 0.47794022],
                          [0.74890292, 0.60624161, 0.26731082],
                          [0.268, 0.570, 0.776]])
-rbd_from_rgb = linalg.inv(rgb_from_rbd)
+# rbd_from_rgb = linalg.inv(rgb_from_rbd)
 
 # Methyl Green + DAB
 rgb_from_gdx = np.array([[0.98003, 0.144316, 0.133146],
                          [0.268, 0.570, 0.776],
                          [0.0, 0.0, 0.0]])
 rgb_from_gdx[2, :] = np.cross(rgb_from_gdx[0, :], rgb_from_gdx[1, :])
-gdx_from_rgb = linalg.inv(rgb_from_gdx)
+# gdx_from_rgb = linalg.inv(rgb_from_gdx)
 
 # Hematoxylin + AEC
 rgb_from_hax = np.array([[0.650, 0.704, 0.286],
                          [0.2743, 0.6796, 0.6803],
                          [0.0, 0.0, 0.0]])
 rgb_from_hax[2, :] = np.cross(rgb_from_hax[0, :], rgb_from_hax[1, :])
-hax_from_rgb = linalg.inv(rgb_from_hax)
+# hax_from_rgb = linalg.inv(rgb_from_hax)
 
 # Blue matrix Anilline Blue + Red matrix Azocarmine + Orange matrix Orange-G
 rgb_from_bro = np.array([[0.853033, 0.508733, 0.112656],
                          [0.09289875, 0.8662008, 0.49098468],
                          [0.10732849, 0.36765403, 0.9237484]])
-bro_from_rgb = linalg.inv(rgb_from_bro)
+# bro_from_rgb = linalg.inv(rgb_from_bro)
 
 # Methyl Blue + Ponceau Fuchsin
 rgb_from_bpx = np.array([[0.7995107, 0.5913521, 0.10528667],
                          [0.09997159, 0.73738605, 0.6680326],
                          [0.0, 0.0, 0.0]])
 rgb_from_bpx[2, :] = np.cross(rgb_from_bpx[0, :], rgb_from_bpx[1, :])
-bpx_from_rgb = linalg.inv(rgb_from_bpx)
+# bpx_from_rgb = linalg.inv(rgb_from_bpx)
 
 # Alcian Blue + Hematoxylin
 rgb_from_ahx = np.array([[0.874622, 0.457711, 0.158256],
                          [0.552556, 0.7544, 0.353744],
                          [0.0, 0.0, 0.0]])
 rgb_from_ahx[2, :] = np.cross(rgb_from_ahx[0, :], rgb_from_ahx[1, :])
-ahx_from_rgb = linalg.inv(rgb_from_ahx)
+# ahx_from_rgb = linalg.inv(rgb_from_ahx)
 
 # Hematoxylin + PAS
 rgb_from_hpx = np.array([[0.644211, 0.716556, 0.266844],
                          [0.175411, 0.972178, 0.154589],
                          [0.0, 0.0, 0.0]])
 rgb_from_hpx[2, :] = np.cross(rgb_from_hpx[0, :], rgb_from_hpx[1, :])
-hpx_from_rgb = linalg.inv(rgb_from_hpx)
+# hpx_from_rgb = linalg.inv(rgb_from_hpx)
 
 # -------------------------------------------------------------
 # The conversion functions that make use of the matrices above
@@ -725,57 +725,57 @@
     return arr @ matrix.T.astype(arr.dtype)
 
 
-@channel_as_last_axis()
-def xyz2rgb(xyz, *, channel_axis=-1):
-    """XYZ to RGB color space conversion.
+# @channel_as_last_axis()
+# def xyz2rgb(xyz, *, channel_axis=-1):
+#     """XYZ to RGB color space conversion.
 
-    Parameters
-    ----------
-    xyz : (..., 3, ...) array_like
-        The image in XYZ format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     xyz : (..., 3, ...) array_like
+#         The image in XYZ format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `xyz` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `xyz` is not at least 2-D with shape (..., 3, ...).
 
-    Notes
-    -----
-    The CIE XYZ color space is derived from the CIE RGB color space. Note
-    however that this function converts to sRGB.
+#     Notes
+#     -----
+#     The CIE XYZ color space is derived from the CIE RGB color space. Note
+#     however that this function converts to sRGB.
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
 
-    Examples
-    --------
-    >>> from skimage import data
-    >>> from skimage.color import rgb2xyz, xyz2rgb
-    >>> img = data.astronaut()
-    >>> img_xyz = rgb2xyz(img)
-    >>> img_rgb = xyz2rgb(img_xyz)
-    """
-    # Follow the algorithm from http://www.easyrgb.com/index.php
-    # except we don't multiply/divide by 100 in the conversion
-    arr = _convert(rgb_from_xyz, xyz)
-    mask = arr > 0.0031308
-    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
-    arr[~mask] *= 12.92
-    np.clip(arr, 0, 1, out=arr)
-    return arr
+#     Examples
+#     --------
+#     >>> from skimage import data
+#     >>> from skimage.color import rgb2xyz, xyz2rgb
+#     >>> img = data.astronaut()
+#     >>> img_xyz = rgb2xyz(img)
+#     >>> img_rgb = xyz2rgb(img_xyz)
+#     """
+#     # Follow the algorithm from http://www.easyrgb.com/index.php
+#     # except we don't multiply/divide by 100 in the conversion
+#     arr = _convert(rgb_from_xyz, xyz)
+#     mask = arr > 0.0031308
+#     arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
+#     arr[~mask] *= 12.92
+#     np.clip(arr, 0, 1, out=arr)
+#     return arr
 
 
 @channel_as_last_axis()
@@ -828,85 +828,85 @@
     return arr @ xyz_from_rgb.T.astype(arr.dtype)
 
 
-@channel_as_last_axis()
-def rgb2rgbcie(rgb, *, channel_axis=-1):
-    """RGB to RGB CIE color space conversion.
+# @channel_as_last_axis()
+# def rgb2rgbcie(rgb, *, channel_axis=-1):
+#     """RGB to RGB CIE color space conversion.
 
-    Parameters
-    ----------
-    rgb : (..., 3, ...) array_like
-        The image in RGB format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     rgb : (..., 3, ...) array_like
+#         The image in RGB format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB CIE format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB CIE format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `rgb` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `rgb` is not at least 2-D with shape (..., 3, ...).
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
 
-    Examples
-    --------
-    >>> from skimage import data
-    >>> from skimage.color import rgb2rgbcie
-    >>> img = data.astronaut()
-    >>> img_rgbcie = rgb2rgbcie(img)
-    """
-    return _convert(rgbcie_from_rgb, rgb)
+#     Examples
+#     --------
+#     >>> from skimage import data
+#     >>> from skimage.color import rgb2rgbcie
+#     >>> img = data.astronaut()
+#     >>> img_rgbcie = rgb2rgbcie(img)
+#     """
+#     return _convert(rgbcie_from_rgb, rgb)
 
 
-@channel_as_last_axis()
-def rgbcie2rgb(rgbcie, *, channel_axis=-1):
-    """RGB CIE to RGB color space conversion.
+# @channel_as_last_axis()
+# def rgbcie2rgb(rgbcie, *, channel_axis=-1):
+#     """RGB CIE to RGB color space conversion.
 
-    Parameters
-    ----------
-    rgbcie : (..., 3, ...) array_like
-        The image in RGB CIE format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     rgbcie : (..., 3, ...) array_like
+#         The image in RGB CIE format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `rgbcie` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `rgbcie` is not at least 2-D with shape (..., 3, ...).
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
 
-    Examples
-    --------
-    >>> from skimage import data
-    >>> from skimage.color import rgb2rgbcie, rgbcie2rgb
-    >>> img = data.astronaut()
-    >>> img_rgbcie = rgb2rgbcie(img)
-    >>> img_rgb = rgbcie2rgb(img_rgbcie)
-    """
-    return _convert(rgb_from_rgbcie, rgbcie)
+#     Examples
+#     --------
+#     >>> from skimage import data
+#     >>> from skimage.color import rgb2rgbcie, rgbcie2rgb
+#     >>> img = data.astronaut()
+#     >>> img_rgbcie = rgb2rgbcie(img)
+#     >>> img_rgb = rgbcie2rgb(img_rgbcie)
+#     """
+#     return _convert(rgb_from_rgbcie, rgbcie)
 
 
 @channel_as_last_axis(multichannel_output=False)
@@ -1253,63 +1253,63 @@
     return xyz2lab(rgb2xyz(rgb), illuminant, observer)
 
 
-@channel_as_last_axis()
-def lab2rgb(lab, illuminant="D65", observer="2", *, channel_axis=-1):
-    """Convert image in CIE-LAB to sRGB color space.
+# @channel_as_last_axis()
+# def lab2rgb(lab, illuminant="D65", observer="2", *, channel_axis=-1):
+#     """Convert image in CIE-LAB to sRGB color space.
 
-    Parameters
-    ----------
-    lab : (..., 3, ...) array_like
-        The input image in CIE-LAB color space.
-        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
-        channels.
-        The L* values range from 0 to 100;
-        the a* and b* values range from -128 to 127.
-    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
-        The name of the illuminant (the function is NOT case sensitive).
-    observer : {"2", "10", "R"}, optional
-        The aperture angle of the observer.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     lab : (..., 3, ...) array_like
+#         The input image in CIE-LAB color space.
+#         Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
+#         channels.
+#         The L* values range from 0 to 100;
+#         the a* and b* values range from -128 to 127.
+#     illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
+#         The name of the illuminant (the function is NOT case sensitive).
+#     observer : {"2", "10", "R"}, optional
+#         The aperture angle of the observer.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in sRGB color space, of same shape as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in sRGB color space, of same shape as input.
 
-    Raises
-    ------
-    ValueError
-        If `lab` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `lab` is not at least 2-D with shape (..., 3, ...).
 
-    Notes
-    -----
-    This function uses :func:`~.lab2xyz` and :func:`~.xyz2rgb`.
-    The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and
-    z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of
-    supported illuminants.
+#     Notes
+#     -----
+#     This function uses :func:`~.lab2xyz` and :func:`~.xyz2rgb`.
+#     The CIE XYZ tristimulus values are x_ref = 95.047, y_ref = 100., and
+#     z_ref = 108.883. See function :func:`~.xyz_tristimulus_values` for a list of
+#     supported illuminants.
 
-    See Also
-    --------
-    rgb2lab
+#     See Also
+#     --------
+#     rgb2lab
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
-    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
-    """
-    xyz, n_invalid = _lab2xyz(lab, illuminant, observer)
-    if n_invalid != 0:
-        warn(
-            "Conversion from CIE-LAB, via XYZ to sRGB color space resulted in "
-            f"{n_invalid} negative Z values that have been clipped to zero",
-            stacklevel=3,
-        )
-    return xyz2rgb(xyz)
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
+#     .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space
+#     """
+#     xyz, n_invalid = _lab2xyz(lab, illuminant, observer)
+#     if n_invalid != 0:
+#         warn(
+#             "Conversion from CIE-LAB, via XYZ to sRGB color space resulted in "
+#             f"{n_invalid} negative Z values that have been clipped to zero",
+#             stacklevel=3,
+#         )
+#     return xyz2rgb(xyz)
 
 
 @channel_as_last_axis()
@@ -1520,74 +1520,74 @@
     return xyz2luv(rgb2xyz(rgb))
 
 
-@channel_as_last_axis()
-def luv2rgb(luv, *, channel_axis=-1):
-    """Luv to RGB color space conversion.
+# @channel_as_last_axis()
+# def luv2rgb(luv, *, channel_axis=-1):
+#     """Luv to RGB color space conversion.
 
-    Parameters
-    ----------
-    luv : (..., 3, ...) array_like
-        The image in CIE Luv format. By default, the final dimension denotes
-        channels.
+#     Parameters
+#     ----------
+#     luv : (..., 3, ...) array_like
+#         The image in CIE Luv format. By default, the final dimension denotes
+#         channels.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `luv` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `luv` is not at least 2-D with shape (..., 3, ...).
 
-    Notes
-    -----
-    This function uses luv2xyz and xyz2rgb.
-    """
-    return xyz2rgb(luv2xyz(luv))
+#     Notes
+#     -----
+#     This function uses luv2xyz and xyz2rgb.
+#     """
+#     return xyz2rgb(luv2xyz(luv))
 
 
-@channel_as_last_axis()
-def rgb2hed(rgb, *, channel_axis=-1):
-    """RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.
+# @channel_as_last_axis()
+# def rgb2hed(rgb, *, channel_axis=-1):
+#     """RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.
 
-    Parameters
-    ----------
-    rgb : (..., 3, ...) array_like
-        The image in RGB format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     rgb : (..., 3, ...) array_like
+#         The image in RGB format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in HED format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in HED format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `rgb` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `rgb` is not at least 2-D with shape (..., 3, ...).
 
-    References
-    ----------
-    .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
-           staining by color deconvolution.," Analytical and quantitative
-           cytology and histology / the International Academy of Cytology [and]
-           American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
+#     References
+#     ----------
+#     .. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
+#            staining by color deconvolution.," Analytical and quantitative
+#            cytology and histology / the International Academy of Cytology [and]
+#            American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
 
-    Examples
-    --------
-    >>> from skimage import data
-    >>> from skimage.color import rgb2hed
-    >>> ihc = data.immunohistochemistry()
-    >>> ihc_hed = rgb2hed(ihc)
-    """
-    return separate_stains(rgb, hed_from_rgb)
+#     Examples
+#     --------
+#     >>> from skimage import data
+#     >>> from skimage.color import rgb2hed
+#     >>> ihc = data.immunohistochemistry()
+#     >>> ihc_hed = rgb2hed(ihc)
+#     """
+#     return separate_stains(rgb, hed_from_rgb)
 
 
 @channel_as_last_axis()
@@ -2115,170 +2115,170 @@
     return arr
 
 
-@channel_as_last_axis()
-def yuv2rgb(yuv, *, channel_axis=-1):
-    """YUV to RGB color space conversion.
+# @channel_as_last_axis()
+# def yuv2rgb(yuv, *, channel_axis=-1):
+#     """YUV to RGB color space conversion.
 
-    Parameters
-    ----------
-    yuv : (..., 3, ...) array_like
-        The image in YUV format. By default, the final dimension denotes
-        channels.
+#     Parameters
+#     ----------
+#     yuv : (..., 3, ...) array_like
+#         The image in YUV format. By default, the final dimension denotes
+#         channels.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `yuv` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `yuv` is not at least 2-D with shape (..., 3, ...).
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/YUV
-    """
-    return _convert(rgb_from_yuv, yuv)
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/YUV
+#     """
+#     return _convert(rgb_from_yuv, yuv)
 
 
-@channel_as_last_axis()
-def yiq2rgb(yiq, *, channel_axis=-1):
-    """YIQ to RGB color space conversion.
+# @channel_as_last_axis()
+# def yiq2rgb(yiq, *, channel_axis=-1):
+#     """YIQ to RGB color space conversion.
 
-    Parameters
-    ----------
-    yiq : (..., 3, ...) array_like
-        The image in YIQ format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     yiq : (..., 3, ...) array_like
+#         The image in YIQ format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `yiq` is not at least 2-D with shape (..., 3, ...).
-    """
-    return _convert(rgb_from_yiq, yiq)
+#     Raises
+#     ------
+#     ValueError
+#         If `yiq` is not at least 2-D with shape (..., 3, ...).
+#     """
+#     return _convert(rgb_from_yiq, yiq)
 
 
-@channel_as_last_axis()
-def ypbpr2rgb(ypbpr, *, channel_axis=-1):
-    """YPbPr to RGB color space conversion.
+# @channel_as_last_axis()
+# def ypbpr2rgb(ypbpr, *, channel_axis=-1):
+#     """YPbPr to RGB color space conversion.
 
-    Parameters
-    ----------
-    ypbpr : (..., 3, ...) array_like
-        The image in YPbPr format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     ypbpr : (..., 3, ...) array_like
+#         The image in YPbPr format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `ypbpr` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `ypbpr` is not at least 2-D with shape (..., 3, ...).
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/YPbPr
-    """
-    return _convert(rgb_from_ypbpr, ypbpr)
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/YPbPr
+#     """
+#     return _convert(rgb_from_ypbpr, ypbpr)
 
 
-@channel_as_last_axis()
-def ycbcr2rgb(ycbcr, *, channel_axis=-1):
-    """YCbCr to RGB color space conversion.
+# @channel_as_last_axis()
+# def ycbcr2rgb(ycbcr, *, channel_axis=-1):
+#     """YCbCr to RGB color space conversion.
 
-    Parameters
-    ----------
-    ycbcr : (..., 3, ...) array_like
-        The image in YCbCr format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     ycbcr : (..., 3, ...) array_like
+#         The image in YCbCr format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `ycbcr` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `ycbcr` is not at least 2-D with shape (..., 3, ...).
 
-    Notes
-    -----
-    Y is between 16 and 235. This is the color space commonly used by video
-    codecs; it is sometimes incorrectly called "YUV".
+#     Notes
+#     -----
+#     Y is between 16 and 235. This is the color space commonly used by video
+#     codecs; it is sometimes incorrectly called "YUV".
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/YCbCr
-    """
-    arr = ycbcr.copy()
-    arr[..., 0] -= 16
-    arr[..., 1] -= 128
-    arr[..., 2] -= 128
-    return _convert(rgb_from_ycbcr, arr)
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/YCbCr
+#     """
+#     arr = ycbcr.copy()
+#     arr[..., 0] -= 16
+#     arr[..., 1] -= 128
+#     arr[..., 2] -= 128
+#     return _convert(rgb_from_ycbcr, arr)
 
 
-@channel_as_last_axis()
-def ydbdr2rgb(ydbdr, *, channel_axis=-1):
-    """YDbDr to RGB color space conversion.
+# @channel_as_last_axis()
+# def ydbdr2rgb(ydbdr, *, channel_axis=-1):
+#     """YDbDr to RGB color space conversion.
 
-    Parameters
-    ----------
-    ydbdr : (..., 3, ...) array_like
-        The image in YDbDr format. By default, the final dimension denotes
-        channels.
-    channel_axis : int, optional
-        This parameter indicates which axis of the array corresponds to
-        channels.
+#     Parameters
+#     ----------
+#     ydbdr : (..., 3, ...) array_like
+#         The image in YDbDr format. By default, the final dimension denotes
+#         channels.
+#     channel_axis : int, optional
+#         This parameter indicates which axis of the array corresponds to
+#         channels.
 
-        .. versionadded:: 0.19
-           ``channel_axis`` was added in 0.19.
+#         .. versionadded:: 0.19
+#            ``channel_axis`` was added in 0.19.
 
-    Returns
-    -------
-    out : (..., 3, ...) ndarray
-        The image in RGB format. Same dimensions as input.
+#     Returns
+#     -------
+#     out : (..., 3, ...) ndarray
+#         The image in RGB format. Same dimensions as input.
 
-    Raises
-    ------
-    ValueError
-        If `ydbdr` is not at least 2-D with shape (..., 3, ...).
+#     Raises
+#     ------
+#     ValueError
+#         If `ydbdr` is not at least 2-D with shape (..., 3, ...).
 
-    Notes
-    -----
-    This is the color space commonly used by video codecs, also called the
-    reversible color transform in JPEG2000.
+#     Notes
+#     -----
+#     This is the color space commonly used by video codecs, also called the
+#     reversible color transform in JPEG2000.
 
-    References
-    ----------
-    .. [1] https://en.wikipedia.org/wiki/YDbDr
-    """
-    return _convert(rgb_from_ydbdr, ydbdr)
+#     References
+#     ----------
+#     .. [1] https://en.wikipedia.org/wiki/YDbDr
+#     """
+#     return _convert(rgb_from_ydbdr, ydbdr)
diff -x *.pyc -bur --co original/package/skimage/segmentation/__init__.py optimized/package/skimage/segmentation/__init__.py
--- original/package/skimage/segmentation/__init__.py	2024-04-26 19:16:30
+++ optimized/package/skimage/segmentation/__init__.py	2024-05-08 16:47:41
@@ -1,11 +1,11 @@
 """Algorithms to partition images into meaningful regions or boundaries.
 """
 
-from ._expand_labels import expand_labels
+# from ._expand_labels import expand_labels
 from .random_walker_segmentation import random_walker
-from .active_contour_model import active_contour
+# from .active_contour_model import active_contour
 from ._felzenszwalb import felzenszwalb
-from .slic_superpixels import slic
+# from .slic_superpixels import slic
 from ._quickshift import quickshift
 from .boundaries import find_boundaries, mark_boundaries
 from ._clear_border import clear_border
@@ -19,11 +19,11 @@
 
 
 __all__ = [
-    'expand_labels',
+    # 'expand_labels',
     'random_walker',
-    'active_contour',
+    # 'active_contour',
     'felzenszwalb',
-    'slic',
+    # 'slic',
     'quickshift',
     'find_boundaries',
     'mark_boundaries',
```