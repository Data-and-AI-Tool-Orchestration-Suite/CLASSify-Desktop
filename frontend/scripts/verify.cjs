/* Verify a route renders the expected Prepare page content. */
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
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await new Promise((r) => setTimeout(r, 600));
    const result = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        dataAndTarget: text.includes("Data & Target"),
        learningMode: text.includes("Learning Mode"),
        randomForest: text.includes("Random Forest"),
        tabpfnHint: text.includes("add-on not installed"),
        startTraining: text.includes("Start Training"),
        actionbar: !!document.querySelector(".prepare-actionbar"),
        advancedSettings: text.includes("Advanced settings"),
        configureColumns: text.includes("Configure Columns"),
      };
    });
    console.log(JSON.stringify(result));
    console.log("pageerrors:", errors.length ? errors : "none");
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
