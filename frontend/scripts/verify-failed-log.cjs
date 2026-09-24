/* Verify failed-run results page shows tabs + log content. */
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
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await new Promise((r) => setTimeout(r, 800));
    const st = await page.evaluate(() => {
      const text = document.body.innerText;
      const tabs = [...document.querySelectorAll(".nav-tabs .nav-link")].map((t) =>
        t.innerText.trim(),
      );
      const logPre = document.querySelector(".nav-tabs")?.parentElement?.querySelector("pre");
      const failureBanner = text.includes("Training failed");
      const openLogBtn = [...document.querySelectorAll("button")].some((b) =>
        b.innerText.includes("Open Output Log"),
      );
      let logHasContent = false;
      let logTabActive = false;
      const active = document.querySelector(".nav-tabs .nav-link.active");
      logTabActive = active ? active.innerText.trim() === "Output Log" : false;
      const pres = [...document.querySelectorAll("pre")];
      logHasContent = pres.some((p) => p.innerText.trim().length > 0);
      return { failureBanner, openLogBtn, tabs, logTabActive, logHasContent };
    });
    console.log(JSON.stringify(st, null, 1));

    // click the banner button → log tab should activate
    await page.evaluate(() => {
      const btn = [...document.querySelectorAll("button")].find((b) =>
        b.innerText.includes("Open Output Log"),
      );
      btn?.click();
    });
    await new Promise((r) => setTimeout(r, 300));
    const after = await page.evaluate(() => {
      const active = document.querySelector(".nav-tabs .nav-link.active");
      return { activeAfterClick: active ? active.innerText.trim() : "none" };
    });
    console.log(JSON.stringify(after));
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
