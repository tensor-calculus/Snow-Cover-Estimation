#set document(
  title: "Your Research Paper Title",
  author: "Your Name",
)

#set page(
  paper: "a4",
  margin: (
    top: 72pt,
    bottom: 72pt,
    left: 72pt,
    right: 72pt,
  ),
  numbering: "1",
)

#set text(
  font: "Times New Roman",
  size: 12pt,
)

#set heading(
  numbering: "1.",
)

#show heading.where(level: 1): set text(
  size: 14pt,
  weight: "bold",
)

#show heading.where(level: 2): set text(
  size: 12pt,
  weight: "bold",
)

#show heading.where(level: 3): set text(
  size: 12pt,
  weight: "bold",
)

#show figure: set block(
  spacing: 12pt,
)

#show figure.caption: set text(
  size: 10pt,
)

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Feasibility Analysis of SAR S-Band for Snow Cover Segmentation
  ]

  #v(12pt)

  R. Virinchi \
  MTech Data Science \
  IIT Guwahati \
]

#v(18pt)

= Introduction

= Literature Survey

// #bibliography(
//   "references.yml",
//   style: "ieee",
// )