use dominator::class;
use once_cell::sync::Lazy;

pub(crate) static MAIN_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("height", "100%")
        .style("display", "flex")
        .style("flex-flow", "column")
        .style("margin-left", "15.1em")
        .style("margin-right", "15.1em")
    }
});

pub(crate) static MENU_DIV_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("margin", "0em")
        .style("list-style-type", "none")
        .style("padding", "0.5em")
        .style("display", "block")
        .style("flex", "0 1 auto")
        .style("background", "#ccf")
    }
});

pub(crate) static SVG_DIV_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("background-color", "#eee")
        .style("outline", "none")
        .style("flex", "1 1 auto")
        .style("overflow", "hidden")
        .style("position", "relative")
    }
});

pub(crate) static LEFT_LEGEND_DIV_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("background-color", "white")
        .style("position", "absolute")
        .style("top", "0")
        .style("left", "0")
        .style("width", "15em")
        .style("font-style", "italic")
        .style("margin-bottom", "1em")
        .style("margin-top", "-.5em")
    }
});

pub(crate) static RIGHT_LEGEND_DIV_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("background-color", "white")
        .style("position", "absolute")
        .style("top", "0")
        .style("right", "0")
        .style("width", "15em")
        .style("font-style", "italic")
        .style("margin-bottom", "1em")
        .style("margin-top", "-.5em")
    }
});

pub(crate) static TITLE_LEGEND_DIV_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("margin", "1px 0 0 0")
        .style("font-size", "100%")
        .style("background-color", "#eef")
        .style("padding", "0.5em")
        .style("font-weight", "normal")
        .style("display", "block")
        .style("margin-block-start", "1em")
        .style("margin-block-end", "1em")
        .style("margin-inline-start", "0px")
        .style("margin-inline-end", "0px")
    }
});

pub(crate) static TEXTAREA_CLASS: Lazy<String> = Lazy::new(|| {
    class! {
        .style("font-family", "monospace")
        .style("width", "100%")
        .style("box-sizing", "border-box")
    }
});

pub(crate) static PATH_CLASS: Lazy<String> = Lazy::new(|| {
    class! {}
});
