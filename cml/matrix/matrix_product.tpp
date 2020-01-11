/* -*- C++ -*- ------------------------------------------------------------
 @@COPYRIGHT@@
 *-----------------------------------------------------------------------*/
/** @file
 */

#ifndef __CML_MATRIX_MATRIX_PRODUCT_TPP
#error "matrix/matrix_product.tpp not included correctly"
#endif

#include <cml/matrix/detail/resize.h>
#include <cml/matrix/fixed_compiled.h>

#include <xmmintrin.h>

namespace cml {

template<class LeftMatrix, class RightMatrix>
using matrix_product_t = matrix_inner_product_promote_t<actual_operand_type_of_t<LeftMatrix>, actual_operand_type_of_t<RightMatrix>>;

template<class LeftMatrix, class RightMatrix,
  enable_if_matrix_t<LeftMatrix>*,
  enable_if_not_fixed_size_t<matrix_traits<LeftMatrix>>*, enable_if_not_row_basis_t<LeftMatrix>*, enable_if_not_row_major_t<LeftMatrix>*,
  enable_if_matrix_rows_greater_t<LeftMatrix, 4>*, enable_if_matrix_cols_greater_t<LeftMatrix, 4>*,
  enable_if_matrix_t<RightMatrix>*,
  enable_if_not_fixed_size_t<matrix_traits<RightMatrix>>*, enable_if_not_row_basis_t<RightMatrix>*, enable_if_not_row_major_t<RightMatrix>*,
  enable_if_matrix_rows_greater_t<RightMatrix, 4>*, enable_if_matrix_cols_greater_t<RightMatrix, 4>*
>
inline auto operator*(LeftMatrix&& left, RightMatrix&& right) -> matrix_product_t<decltype(left), decltype(right)>
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

template<class LeftMatrix, class RightMatrix,
  enable_if_matrix_t<LeftMatrix>*,
  enable_if_fixed_size_t<matrix_traits<LeftMatrix>>*, enable_if_row_basis_t<LeftMatrix>*, enable_if_row_major_t<LeftMatrix>*,
  enable_if_matrix_rows_less_or_equal_t<LeftMatrix, 4>*, enable_if_matrix_cols_less_or_equal_t<LeftMatrix, 4>*,
  enable_if_matrix_t<RightMatrix>*,
  enable_if_fixed_size_t<matrix_traits<RightMatrix>>*, enable_if_row_basis_t<RightMatrix>*, enable_if_row_major_t<RightMatrix>*,
  enable_if_matrix_rows_less_or_equal_t<RightMatrix, 4>*, enable_if_matrix_cols_less_or_equal_t<RightMatrix, 4>*
>
inline auto operator*(LeftMatrix&& left, RightMatrix&& right) -> matrix_product_t<decltype(left), decltype(right)>
{
  typedef matrix_inner_product_promote_t<
    actual_operand_type_of_t<decltype(left)>,
    actual_operand_type_of_t<decltype(right)>> result_type;

  cml::check_same_inner_size(left, right);

  result_type  result = { };
  float const* p_left_row = left.data();
  float*       p_result_row = result.data();

  std::array<__m128, matrix_traits<RightMatrix>::storage_type::array_cols> right_cols;
  for (int col = 0; col < right.cols(); ++col) {
    right_cols[col] = _mm_loadu_ps(right.data() + col * right.rows());
  }

  for (int row = 0; row < left.rows(); ++row, p_left_row += left.cols(), p_result_row += result.cols()) {
    __m128 res_row = _mm_setzero_ps();
    for (int col = 0; col < left.cols(); ++col) {
        __m128 left_element = _mm_set1_ps(p_left_row[col]);
        res_row  = _mm_add_ps(res_row, _mm_mul_ps(left_element, right_cols[col]));
    }
    _mm_storeu_ps(p_result_row, res_row);
  }

  return result;
}

} // namespace cml

// -------------------------------------------------------------------------
// vim:ft=cpp:sw=2
