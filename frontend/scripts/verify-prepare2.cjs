/* Verify prepare-page model selection + toast position. */
const puppeteer = require("puppeteer-core");

(async () => {
  const url = process.argv[2];
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    headless: "new",
    args: ["--disable-gpu", "--no-first-run"],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1750 });
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await new Promise((r) => setTimeout(r, 600));
    const st = await page.evaluate(() => {
      const tabpfn = [...document.querySelectorAll(".model-card")].find((c) =>
        c.textContent.includes("TabPFN"),
      );
      const rf = [...document.querySelectorAll(".model-card")].find((c) =>
        c.textContent.includes("Random Forest"),
      );
      const toastContainer = document.querySelector(".toast-container");
      const ts = getComputedStyle(toastContainer);
      return {
        tabpfnChecked: tabpfn.querySelector("input").checked,
        tabpfnDisabled: tabpfn.querySelector("input").disabled,
        rfChecked: rf.querySelector("input").checked,
        toastPos: `${ts.position} top=${ts.top} right=${ts.right} bottom=${ts.bottom}`,
        toastZ: ts.zIndex,
      };
    });
    console.log(JSON.stringify(st));

    // toast visibility check: fire one via the app? simulate by injecting
    await page.evaluate(() => {
      const c = document.querySelector(".toast-container");
      c.classList.remove("bottom-0");
      const el = document.elementFromPoint(window.innerWidth - 30, window.innerHeight - 30);
      return el ? el.tagName : "none";
    });

    // click the whole tabpfn card to prove it cannot be selected
    await page.evaluate(() => {
      const tabpfn = [...document.querySelectorAll(".model-card")].find((c) =>
        c.textContent.includes("TabPFN"),
      );
      tabpfn.click();
    });
    await new Promise((r) => setTimeout(r, 300));
    const still = await page.evaluate(() => {
      const tabpfn = [...document.querySelectorAll(".model-card")].find((c) =>
        c.textContent.includes("TabPFN"),
      );
      return { stillUnchecked: !tabpfn.querySelector("input").checked };
    });
    console.log(JSON.stringify(still));
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
