/* Capture a route of the running app with puppeteer-core. */
const puppeteer = require("puppeteer-core");

const url = process.argv[2];
const out = process.argv[3];
const width = Number.parseInt(process.argv[4] || "1440", 10);
const height = Number.parseInt(process.argv[5] || "1200", 10);

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    headless: "new",
    args: ["--disable-gpu", "--no-first-run"],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width, height });
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await new Promise((r) => setTimeout(r, 800));
    await page.screenshot({ path: out });
    console.log("saved", out);
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
