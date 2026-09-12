#set page(
  paper: "presentation-16-9",
  margin: (x: 2.5cm, y: 2cm),
  footer: counter(page).display("1 / 1", both: true),
)

#set text(
  font: "Liberation Sans", 
  size: 22pt,
  fill: black,
)

// --- Slide 1: Title ---
#align(center + horizon)[
  #text(32pt, weight: "bold")[Minimal Text Presentation]

  #v(1em)
  #text(20pt)[Subtitle or Author]
]

#pagebreak()

// --- Slide 2 ---
== Core Objectives

- Zero visual clutter
- No underlines, colored banners, or progress bars
- Clean black and white text

#pagebreak()

// --- Slide 3 ---
== Layout Control

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    *Column One*
    - Simple text items
    - Fully scannable
  ],
  [
    *Column Two*
    - Standard layout
    - Plain typography
  ]
)

#pagebreak()

// --- Slide 4 ---
== Final Notes

- Just write standard Markdown headings (`== Slide Title`)
- Use `#pagebreak()` to start a new slide