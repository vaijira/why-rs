use dominator::class;
use once_cell::sync::Lazy;

pub(crate) static MAIN_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("height", "100%")
        .style("display", "flex")
        .style("flex-flow", "column")
    }
});

pub(crate) static SVG_DIV_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("background-color", "#eee")
        .style("outline", "none")
        .style("flex", "1 1 auto")
        .style("overflow", "hidden")
    }
});

pub(crate) static PATH_CLASS: Lazy<String> = Lazy::new(|| {
    class! {}
});
