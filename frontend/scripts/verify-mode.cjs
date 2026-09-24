/* Verify learning-mode switch on Prepare + capture pages. */
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

    const before = await page.evaluate(() =>
      document.body.innerText.includes("Spectral Clustering"),
    );
    await page.click('label[for="unsupervised"]');
    await new Promise((r) => setTimeout(r, 400));
    const after = await page.evaluate(() => document.body.innerText);
    const result = {
      spectralBeforeSwitch: before,
      spectralAfterSwitch: after.includes("Spectral Clustering"),
      randomForestAfterSwitch: after.includes("Random Forest"),
      kmeansBadgeGone: !after.includes("num_clusters") || after.includes("Number of clusters"),
    };
    console.log(JSON.stringify(result));
    console.log("pageerrors:", errors.length ? errors : "none");
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
