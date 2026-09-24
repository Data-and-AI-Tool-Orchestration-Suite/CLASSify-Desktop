/* Verify navbar logo + favicon wiring on the running app. */
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
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    const st = await page.evaluate(() => ({
      navbarImg: !!document.querySelector(".navbar-brand img.brand-logo"),
      imgLoaded: (document.querySelector(".navbar-brand img.brand-logo")?.naturalWidth ?? 0) > 0,
      faviconLink: !!document.querySelector('link[rel="icon"][href="/favicon.png"]'),
    }));
    console.log(JSON.stringify(st));
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
