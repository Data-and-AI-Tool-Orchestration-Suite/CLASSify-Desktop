import preprocess from "svelte-preprocess";

export default {
  preprocess: preprocess({
    scss: { includePaths: ["src", "static/css"] },
  }),
};
