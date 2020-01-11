/* -*- C++ -*- ------------------------------------------------------------
 @@COPYRIGHT@@
 *-----------------------------------------------------------------------*/
/** @file
 */

#pragma once

#ifndef	cml_common_layout_tags_h
#define	cml_common_layout_tags_h

#include <cml/common/traits.h>

namespace cml {

/* Forward declarations: */
struct row_major;
struct col_major;
struct any_major;

enum layout_kind {
  row_major_c = 1,
  col_major_c = 2,
  any_major_c = 3,
  layout_count = 3
};

/** Row major tag. */
struct row_major {

  /** row_major transposes to col_major. */
  typedef col_major					transposed_tag;

  /** Integral identifier. */
  static const layout_kind value = row_major_c;
};

/** Column major tag. */
struct col_major {

  /** col_major transposes to row_major. */
  typedef row_major					transposed_tag;

  /** Integral identifier. */
  static const layout_kind value = col_major_c;
};

/** Arbitrary or unspecified major tag. */
struct any_major {

  /** any_major transposes to itself. */
  typedef any_major					transposed_tag;

  /** Integral identifier. */
  static const layout_kind value = any_major_c;
};


/** Detect valid layout tags.
 *
 * @note This can be specialized for user-defined layout tags.
 */
template<class Tag> struct is_layout_tag {
  static const bool value
    =  std::is_same<Tag, row_major>::value
    || std::is_same<Tag, col_major>::value
    || std::is_same<Tag, any_major>::value;
};

/** Templated helper to determine the layout tag of an expression that
 * defines the layout_tag type.
 */
template<class T> struct layout_tag_of {
  typedef typename T::layout_tag type;
  static_assert(is_layout_tag<type>::value, "invalid layout tag");
};

/** Convenience alias for layout_tag_of. */
template<class T> using layout_tag_of_t = typename layout_tag_of<T>::type;

/** Retrieve the layout_tag of @c T via traits. */
template<class T> struct layout_tag_trait_of {
  typedef typename traits_of<T>::type::layout_tag type;
  static_assert(is_layout_tag<type>::value, "invalid layout tag");
};

/** Convenience alias for layout_tag_trait_of. */
template<class T>
  using layout_tag_trait_of_t = typename layout_tag_trait_of<T>::type;

/** Helper to detect row major layout types. */
template<class T> struct is_row_major {
  static const bool value = std::is_same<layout_tag_trait_of_t<T>, row_major>::value;
};

/** Wrapper for enable_if to detect types tagged with row_major. */
template<class Sub, class T = void> struct enable_if_row_major
: std::enable_if<is_row_major<Sub>::value, T> {};

/** Wrapper for enable_if to detect types not tagged with row_major. */
template<class Sub, class T = void> struct enable_if_not_row_major
: std::enable_if<!is_row_major<Sub>::value, T> {};

/** Convenience alias for enable_if_row_major. */
template<class Sub, class T = void> using enable_if_row_major_t
= typename enable_if_row_major<Sub, T>::type;

/** Convenience alias for enable_if_not_row_major. */
template<class Sub, class T = void> using enable_if_not_row_major_t
= typename enable_if_not_row_major<Sub, T>::type;

/** Helper to detect column major layout types. */
template<class T> struct is_col_major {
  static const bool value = std::is_same<layout_tag_trait_of_t<T>, col_major>::value;
};

/** Wrapper for enable_if to detect types tagged with col_major. */
template<class Sub, class T = void> struct enable_if_col_major
: std::enable_if<is_col_major<Sub>::value, T> {};

/** Wrapper for enable_if to detect types not tagged with col_major. */
template<class Sub, class T = void> struct enable_if_not_col_major
: std::enable_if<!is_col_major<Sub>::value, T> {};

/** Convenience alias for enable_if_col_major. */
template<class Sub, class T = void> using enable_if_col_major_t
= typename enable_if_col_major<Sub, T>::type;

/** Convenience alias for enable_if_not_col_major. */
template<class Sub, class T = void> using enable_if_not_col_major_t
= typename enable_if_not_col_major<Sub, T>::type;

} // namespace cml

#endif

// -------------------------------------------------------------------------
// vim:ft=cpp:sw=2
