/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_CONFIG_H_IN
#define CK_CONFIG_H_IN

// clang-format off
//
// DataType supports in the current CK build
//
#ifndef DTYPES
/* #undef DTYPES */
#endif
// if DTYPES is not defined, enable all datatypes in headerfiles
#ifndef CK_ENABLE_ALL_DTYPES
/* #undef CK_ENABLE_ALL_DTYPES */
#if defined(CK_ENABLE_ALL_DTYPES)
#ifndef CK_ENABLE_INT8
#define CK_ENABLE_INT8 "ON"
#endif
#ifndef CK_ENABLE_FP8
#define CK_ENABLE_FP8 "ON"
#endif
#ifndef CK_ENABLE_BF8
#define CK_ENABLE_BF8 "ON"
#endif
#ifndef CK_ENABLE_FP16
#define CK_ENABLE_FP16 "ON"
#endif
#ifndef CK_ENABLE_BF16
#define CK_ENABLE_BF16 "ON"
#endif
#ifndef CK_ENABLE_FP32
#define CK_ENABLE_FP32 "ON"
#endif
#ifndef CK_ENABLE_FP64
#define CK_ENABLE_FP64 "ON"
#endif
#endif
#endif
// if DTYPES are selectively enabled
#ifndef CK_ENABLE_INT8
#define CK_ENABLE_INT8 ON
#endif

#ifndef CK_ENABLE_FP8
#define CK_ENABLE_FP8 ON
#endif

#ifndef CK_ENABLE_BF8
#define CK_ENABLE_BF8 ON
#endif

#ifndef CK_ENABLE_FP16
#define CK_ENABLE_FP16 ON
#endif

#ifndef CK_ENABLE_BF16
#define CK_ENABLE_BF16 ON
#endif

#ifndef CK_ENABLE_FP32
#define CK_ENABLE_FP32 ON
#endif

#ifndef CK_ENABLE_FP64
#define CK_ENABLE_FP64 ON
#endif

//
// Legacy DL kernel supports in the current CK build
// by default DL kernels are turned OFF
//
#ifndef CK_ENABLE_DL_KERNELS
#define CK_ENABLE_DL_KERNELS ON
#endif

#ifndef CK_ENABLE_DPP_KERNELS
#define CK_ENABLE_DPP_KERNELS ON
#endif

//
// CK kernels which support XDL (MI series)
//
#ifndef CK_USE_XDL
/* #undef CK_USE_XDL */
#endif

//
// CK Kernels which support WMMA (recent Navi series)
//
#ifndef CK_USE_WMMA
/* #undef CK_USE_WMMA */
#endif

#ifndef CK_USE_WMMA_FP8
/* #undef CK_USE_WMMA_FP8 */
#endif

#ifndef CK_USE_GFX94
/* #undef CK_USE_GFX94 */
#endif

#ifndef CK_USE_OCP_FP8
/* #undef CK_USE_OCP_FP8 */
#endif

#ifndef CK_USE_FNUZ_FP8
/* #undef CK_USE_FNUZ_FP8 */
#endif

#ifndef CK_USE_FP8_ON_UNSUPPORTED_ARCH
/* #undef CK_USE_FP8_ON_UNSUPPORTED_ARCH */
#endif

#ifndef CK_USE_NATIVE_MX_SUPPORT
/* #undef CK_USE_NATIVE_MX_SUPPORT */
#endif

// clang-format on

#endif // CK_CONFIG_H_IN
