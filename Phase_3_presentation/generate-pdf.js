const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const PptxGenJS = require('pptxgenjs');

async function generatePPTX() {
    console.log('Starting PPTX generation...\n');

    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();

    // Set viewport to 1280x720 (the design size) with higher DPI for quality
    await page.setViewport({
        width: 1280,
        height: 720,
        deviceScaleFactor: 2  // Higher resolution for crisp images
    });

    // Create a new PPTX presentation (widescreen 16:9)
    const pptx = new PptxGenJS();
    pptx.layout = 'LAYOUT_WIDE';  // 13.33" x 7.5" (16:9)

    const TOTAL_SLIDES = 12;

    for (let i = 1; i <= TOTAL_SLIDES; i++) {
        const htmlFile = path.join(__dirname, `index${i}.html`);

        console.log(`Processing slide ${i}/${TOTAL_SLIDES}...`);

        // Open the HTML file
        await page.goto(`file://${htmlFile}`, {
            waitUntil: 'networkidle0',
            timeout: 30000
        });

        // Wait for fonts and images to load
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Take screenshot as PNG buffer
        const screenshot = await page.screenshot({
            type: 'png',
            fullPage: false,
            clip: { x: 0, y: 0, width: 1280, height: 720 }
        });

        // Convert buffer to base64 data URI for pptxgenjs
        const base64 = screenshot.toString('base64');
        const dataUri = `data:image/png;base64,${base64}`;

        // Add a slide and place the screenshot as a full-bleed background image
        const slide = pptx.addSlide();
        slide.addImage({
            data: dataUri,
            x: 0,
            y: 0,
            w: '100%',
            h: '100%',
        });

        console.log(`  ✓ Slide ${i} captured`);
    }

    await browser.close();

    // Write the PPTX file
    const outputPath = path.join(__dirname, 'presentation.pptx');
    await pptx.writeFile({ fileName: outputPath });

    console.log(`\n✅ PPTX generated successfully: presentation.pptx`);
    console.log('   Layout: Widescreen 16:9 (13.33" × 7.5")');
}

generatePPTX().catch(console.error);
