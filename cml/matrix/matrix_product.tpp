/* -*- C++ -*- ------------------------------------------------------------
 @@COPYRIGHT@@
 *-----------------------------------------------------------------------*/
/** @file
 */

#ifndef __CML_MATRIX_MATRIX_PRODUCT_TPP
#error "matrix/matrix_product.tpp not included correctly"
#endif

#include <cml/matrix/detail/resize.h>
#include <cml/matrix/types.h>

#include <xmmintrin.h>

namespace cml {

template<class Sub1, class Sub2>
using matrix_product_t = matrix_inner_product_promote_t<actual_operand_type_of_t<Sub1>, actual_operand_type_of_t<Sub2>>;

template<class Sub1, class Sub2, enable_if_matrix_t<Sub1>*, enable_if_matrix_t<Sub2>*>
inline auto operator*(Sub1&& left, Sub2&& right) -> matrix_product_t<decltype(left), decltype(right)>
{
  cml::check_same_inner_size(left, right);

  matrix_product_t<decltype(left), decltype(right)> M;
  detail::resize(M, array_rows_of(left), array_cols_of(right));
  for(int i = 0; i < M.rows(); ++ i) {
    for(int j = 0; j < M.cols(); ++ j) {
      auto m = left(i,0) * right(0,j);
      for(int k = 1; k < left.cols(); ++ k) m += left(i,k) * right(k,j);
      M(i,j) = m;
    }
  }
  return M;
}

// SSE Optimized matrix product specialization for float fixed matrices with row major alignment and row basis

template<int Rows, int Cols>
using matrixRCf_r = matrix<float, fixed<Rows, Cols>, row_basis, row_major>;

template<int Dim>
using product_matrixRCf_r = matrix<float, fixed<Dim, Dim>, row_basis, row_major>;

template<int Rows, int Cols, class LeftMatrix = matrixRCf_r<Rows, Cols>, class RightMatrix = matrixRCf_r<Cols, Rows>>
inline product_matrixRCf_r<Rows> multiply_matrix_float_fixed_row_order(LeftMatrix&& left, RightMatrix&& right)
{
  product_matrixRCf_r<Rows> result;
  float const* p_left_row   = left.data();
  float*       p_result_row = result.data();

  std::array<__m128, Rows> right_cols;
  for (int col = 0; col < right.cols(); ++col)
  {
    right_cols[col] = _mm_loadu_ps(right.data() + col * right.rows());
  }

  for (int row = 0; row < left.rows(); ++row, p_left_row += left.cols(), p_result_row += result.cols())
  {
    __m128 res_row = _mm_setzero_ps();
    for (int col = 0; col < left.cols(); ++col)
    {
      __m128 left_element = _mm_set1_ps(p_left_row[col]);
      res_row  = _mm_add_ps(res_row, _mm_mul_ps(left_element, right_cols[col]));
    }
    _mm_storeu_ps(p_result_row, res_row);
  }

  return result;
}

template<>
inline matrix44f_r operator*<>(matrix44f_r&& left, matrix44f_r&& right)
{
    return multiply_matrix_float_fixed_row_order<4, 4>(left, right);
}

template<>
inline matrix44f_r operator*<>(matrix44f_r&& left, const matrix44f_r& right)
{
    return multiply_matrix_float_fixed_row_order<4, 4>(left, right);
}

template<>
inline matrix44f_r operator*<>(matrix44f_r&& left, matrix44f_r& right)
{
    return multiply_matrix_float_fixed_row_order<4, 4>(left, right);
}

template<>
inline matrix44f_r operator*<>(const matrix44f_r& left, const matrix44f_r& right)
{
    return multiply_matrix_float_fixed_row_order<4, 4>(left, right);
}

template<>
inline matrix44f_r operator*<>(matrix44f_r& left, const matrix44f_r& right)
{
    return multiply_matrix_float_fixed_row_order<4, 4>(left, right);
}

template<>
inline matrix44f_r operator*<>(const matrix44f_r& left, matrix44f_r& right)
{
    return multiply_matrix_float_fixed_row_order<4, 4>(left, right);
}

template<>
inline matrix44f_r operator*<>(matrix44f_r& left, matrix44f_r& right)
{
    return multiply_matrix_float_fixed_row_order<4, 4>(left, right);
}

template<>
inline matrix33f_r operator*<>(const matrix33f_r& left, const matrix33f_r& right)
{
    return multiply_matrix_float_fixed_row_order<3, 3>(left, right);
}

template<>
inline matrix33f_r operator*<>(matrix33f_r& left, const matrix33f_r& right)
{
    return multiply_matrix_float_fixed_row_order<3, 3>(left, right);
}

template<>
inline matrix33f_r operator*<>(const matrix33f_r& left, matrix33f_r& right)
{
    return multiply_matrix_float_fixed_row_order<3, 3>(left, right);
}

template<>
inline matrix33f_r operator*<>(matrix33f_r& left, matrix33f_r& right)
{
    return multiply_matrix_float_fixed_row_order<3, 3>(left, right);
}

} // namespace cml

// -------------------------------------------------------------------------
// vim:ft=cpp:sw=2
