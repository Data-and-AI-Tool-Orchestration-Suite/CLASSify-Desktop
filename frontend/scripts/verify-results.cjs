/* Verify results table expand behavior. */
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
    await page.setViewport({ width: 1440, height: 1400 });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await page.waitForSelector("table tbody tr", { timeout: 15000 });
    await new Promise((r) => setTimeout(r, 500));

    const rowCount = await page.evaluate(() => document.querySelectorAll("tbody > tr").length);
    await page.click("tbody button.btn-link");
    await new Promise((r) => setTimeout(r, 400));
    const after = await page.evaluate(() => {
      const rows = [...document.querySelectorAll("tbody > tr")];
      const detail = rows.find((r) => r.querySelector(".badge.text-bg-light"));
      return {
        rowsAfterClick: rows.length,
        detailRowPresent: !!detail,
        chips: detail ? detail.querySelectorAll(".badge.text-bg-light").length : 0,
        metricTiles: detail ? detail.querySelectorAll(".border.rounded").length : 0,
        firstRowHeight: Math.round(rows[0].getBoundingClientRect().height),
      };
    });
    console.log(JSON.stringify({ rowCount, ...after }));
    console.log("errs:", errors.length ? errors : "none");
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
