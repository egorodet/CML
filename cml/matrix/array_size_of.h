/* -*- C++ -*- ------------------------------------------------------------
 @@COPYRIGHT@@
 *-----------------------------------------------------------------------*/
/** @file
 */

#pragma once

#ifndef	cml_matrix_array_size_of_h
#define	cml_matrix_array_size_of_h

#include <cml/common/array_size_of.h>
#include <cml/vector/type_util.h>

namespace cml {

/** Specialize array_rows_of_c for vectors to return 1. */
template<class Sub>
struct array_rows_of_c<Sub, enable_if_vector_t<Sub>> {
  static const int value = 1;
};

/** Specialize array_cols_of_c for vectors to return 1. */
template<class Sub>
struct array_cols_of_c<Sub, enable_if_vector_t<Sub>> {
  static const int value = 1;
};

/** Convenience alias to detect matrices with rows count less than N. */
template<class Sub, size_t N, class T = void>
using enable_if_matrix_rows_less_or_equal
= std::enable_if<matrix_traits<Sub>::storage_type::array_rows <= N, T>;

/** Convenience alias for enable_if_matrix_rows_less_or_equal. */
template<class Sub, size_t N, class T = void> using enable_if_matrix_rows_less_or_equal_t
= typename enable_if_matrix_rows_less_or_equal<Sub, N, T>::type;

/** Convenience alias to detect matrices with rows count is greater than N. */
template<class Sub, size_t N, class T = void>
using enable_if_matrix_rows_greater
= std::enable_if<!(matrix_traits<Sub>::storage_type::array_rows <= N), T>;

/** Convenience alias for enable_if_matrix_rows_greater. */
template<class Sub, size_t N, class T = void> using enable_if_matrix_rows_greater_t
= typename enable_if_matrix_rows_greater<Sub, N, T>::type;

/** Convenience alias to detect matrices with cols count less than N. */
template<class Sub, size_t N, class T = void>
using enable_if_matrix_cols_less_or_equal
= std::enable_if<matrix_traits<Sub>::storage_type::array_cols <= N, T>;

/** Convenience alias for enable_if_matrix_cols_less_or_equal. */
template<class Sub, size_t N, class T = void> using enable_if_matrix_cols_less_or_equal_t
= typename enable_if_matrix_cols_less_or_equal<Sub, N, T>::type;

/** Convenience alias to detect matrices with cols count is greater than N. */
template<class Sub, size_t N, class T = void>
using enable_if_matrix_cols_greater
= std::enable_if<!(matrix_traits<Sub>::storage_type::array_cols <= N), T>;

/** Convenience alias for enable_if_matrix_cols_greater. */
template<class Sub, size_t N, class T = void> using enable_if_matrix_cols_greater_t
= typename enable_if_matrix_cols_greater<Sub, N, T>::type;


} // namespace cml

#endif

// -------------------------------------------------------------------------
// vim:ft=cpp:sw=2
