<!-- Parts of the planned configuration (Raspberry Pi 5 + MCC 128). Web survey of 2026-11-03 (mostly search snippets; re-check before buying). -->

# Planned upgrade: parts price survey (checked 2026-11-03)

Prices are tax-included (10%) unless marked. Where the source shows the price without tax, it is converted at x1.10. USD is converted at 1 USD = 150 JPY. Shipping fees are not included unless stated.

| # | item | model | qty | unit price JPY | tax incl.? | source URL | note |
|---|---|---|---|---|---|---|---|
| 1 | Raspberry Pi 5 8GB | RPI-SC1112 | 1 | 36,300 | yes | https://www.switch-science.com/products/9250 | Price went up because of the 2025-26 DRAM price spike (launch price was 15,290). Switch Science says it will revise prices periodically. |
| 2 | Official 27 W USB-C PSU | Raspberry Pi SC1153 | 1 | 2,176 (1,978 excl.) | both shown | https://www.digikey.jp/ja/products/detail/raspberry-pi/SC1153/21658276 | Japanese-market alternative: Switch Science's own 5.1 V/5 A PD adapter (SSCI-093453, PSE marked) at 2,387 incl.: https://www.switch-science.com/products/9345 |
| 3a | Pi 5 Active Cooler | RPI-SC1148 | 1 | 1,100 | yes | https://www.switch-science.com/products/9253 | Price from search snippets of the page (one older snippet showed 1,023). |
| 3b | Tall 2x20 pin socket (GPIO riser) | SparkFun PRT-14017 | 1 | 726 | yes | https://www.switch-science.com/products/3076 | Check the height fits over the Active Cooler. |
| 3c | Brass spacer kit (M2.5/M3) | SparkFun KIT-24115 | 1 | 4,290 | yes | https://www.switch-science.com/products/9457 (search snippet) | More than needed. 4 x M2.5 standoffs from RS/MonotaRO cost a few hundred yen. Check whether the MCC 128 box already includes standoffs. |
| 4 | MCC 128 DAQ HAT | Digilent/MCC 6069-410-001 | 1 | 44,453 (40,412 excl.) | both shown | https://www.digikey.jp/ja/products/detail/digilent-inc/6069-410-001/15822144 | Digi-Key JP stock 0, lead time 18 weeks. US street price is $199 (T-Equipment) to $199.99 (RobotShop), about 29,850-30,000 JPY before shipping and import tax: https://tequipment.net/Digilent/series_mcc-daq-hat |
| 5a | Raspberry Pi Monitor 15.6" | RPI-SC0940 (red/white) | 1 | 22,550 | yes | https://www.switch-science.com/products/10001 | 20 in stock |
| 5b | Official micro-HDMI to HDMI cable 1 m | SC0358 | 1 | 1,100 | yes | https://www.switch-science.com/products/8626 | 2 m version 1,320 (products/10311) |
| 6 | Portable SSD 500 GB | SanDisk Extreme V2 SDSSDE61-500G-J25 | 1 | 31,680 | yes (likely) | https://kakaku.com/item/K0001325952/ | Kakaku.com lowest price (Rakuten). One search snippet showed 28,800 (Amazon.co.jp). SSD prices are inflated in 2026 (NAND spike). A 256 GB or cheaper brand would cost less. Samsung T7 500GB had no current JP listing. |
| 7a | Absolute pressure sensor 15-115 kPa | NXP MPXA6115A6U (SSOP-8) | 1 | 4,587 (4,170 excl.) | both shown | https://www.digikey.jp/ja/products/result?keywords=MPXA6115A6U | Active, 758 in stock. MPX4115A is listed "No longer manufactured" (element14). Surface-mount, so it needs a breakout or the custom PCB. |
| 7b | 5 V regulator 100 mA | Toshiba TA78L05S | 2 | 20 | yes | https://akizukidenshi.com/catalog/g/g108973/ | L78L05ACZ 5 pcs for 100 (g101250) |
| 8a | Photocoupler | Toshiba TLP785(GB) | 2 | 20 | yes | https://akizukidenshi.com/catalog/g/g107554/ | In stock |
| 8b | NPN transistor | 2SC1815 (20-pc pack) | 1 pack | 100 | yes | https://akizukidenshi.com/catalog/g/g117089/ | The original Toshiba 2SC1815 is discontinued. Akizuki sells 2SC1815 and 2SC1815L-GR/Y/BL equivalents (UTC). |
| 8c | Relay DPDT 24 VDC | OMRON G6A-274P-ST-US DC24 | 2 | 1,251 (1,137 excl.) | both shown | https://www.digikey.jp/ja/products/result?keywords=G6A-274P-ST-US%20DC24 | Active, about 10k in stock (Digi-Key part 39-G6A-274P-ST-USDC24-ND) |
| 8d | Diode | 1N4148 (50-pc pack) | 1 pack | 200 | yes | https://akizukidenshi.com/catalog/g/g100941/ | |
| 8e | Resistors 1/4 W carbon (100-pc bag per value) | CF25J1KB etc. | 2 bags | 100 | yes | https://akizukidenshi.com/catalog/g/g125102/ | 100 yen per 100-pc bag per value |
| 8f | "Relay sockets" | round-pin single-row socket strip 1x20 (6604S-20) | 2 | 80 | yes | https://akizukidenshi.com/catalog/g/g100241/ | No dedicated OMRON socket for G6A was found (P6B is for G6B). Use cut round-pin strips as sockets, or solder the relay directly. |
| 9a | Op-amp | TI TL071CP (PDIP-8) | 1 | 92 (84 excl.) | both shown | https://www.digikey.jp/ja/products/detail/texas-instruments/TL071CP/378372 | Active, 3,185 in stock |
| 9b | DIP-8 socket (round pin) | 2227MC-08-03 | 1 | 20 | yes | https://akizukidenshi.com/catalog/g/g100035/ | |
| 10 | Suppression parts (lump sum) | 0.1 uF 50 V X7R x10 (100), TVS P4KE15CA x10 (180), clamp ferrite e.g. LF-35B x4 (110 each) | 1 lot | 720 | yes | https://akizukidenshi.com/catalog/g/g113582/ , https://akizukidenshi.com/catalog/g/g129635/ , https://akizukidenshi.com/catalog/g/g103748/ | The Akizuki TVS search showed only 6.8/15 V P4KE and 45 V SMAJ/SMBJ parts. A TVS for a 24 V line (about 30-33 V bidirectional, e.g. P6KE33CA) costs about the same per piece but was not priced. |
| 11 | Fuses + holders (lump sum) | Panel holder MF-525M (midget) x3 (180 each) + 20 mm glass fuses x6 (40 each) | 1 lot | 780 | yes | https://akizukidenshi.com/catalog/g/g107080/ , https://akizukidenshi.com/catalog/g/g107129/ | PCB clip FUC-03A is 20 yen each (g110521) |
| 12 | Custom PCB, 2-layer, ≤100x100 mm, 5 pcs | JLCPCB | 1 lot | about 2,850 | n/a | https://jlcpcb.com/help/article/pcb-fabrication-services-and-production-time ; DHL figure: https://eevblog.com/forum/manufacture/jlcpcb-offer-until-25-december/msg3336954 | $2 board + about $17 DHL (forum report, not Japan-specific) = about $19 at 150 JPY/USD. The exact shipping cost appears only at checkout. |
| 13 | Safety cover sheet | Polycarbonate sheet clear 2 mm, 1000x500 (ESCO EA441PE series) | 2 | 6,010 | yes | https://jp.misumi-ec.com/vona2/detail/223007095468/ (search snippet only, page not opened) | Two sheets are enough for about a 30 cm box (5 faces). Add brackets and screws. A ready-made clear PC box of this size was not found. AS ONE Bopla PC case 300x230x110 is grey, 24,100 excl. |
| 14 | In-line water trap / filter, pump inlet | (e.g. SMC ZFC33 in-line vacuum filter, -100 kPa rated) | 1 | not priced | - | https://uk.rs-online.com/web/p/pneumatic-filters/2887015 | No JPY price found. UK RS lists GBP 7.18 ex VAT. A particulate filter is not a liquid trap; for water a trap bottle may be needed in front of it. |
| 15 | Vacuum limiting for about -50 kPa | SMC IRV10 vacuum regulator (functional substitute) | 1 | from 8,468 | yes | https://www.monotaro.com/s/c-137676 (search snippet only; 7,698 excl.) | A true vacuum relief valve exists (Yoshitake VB-7, spring C -48 to -81 kPa: https://www.yoshitake.co.jp/upload/up_file/products/VB-7_catalog_en.pdf) but no price was found. MISUMI VR100 (-6.5 to -100 kPa) price not confirmed. |

## Subtotals (tax incl., shipping excluded)

| group | items | subtotal JPY |
|---|---|---|
| (a) computer + DAQ + display | 1-6 | **144,375** (36,300 + 2,176 + 6,116 + 44,453 + 23,650 + 31,680) |
| (b) sensor and circuit parts | 7-12 | **12,291** (4,627 + 3,202 + 112 + 720 + 780 + 2,850) |
| (c) safety / maintenance | 13-15 | **20,488** (12,020 + 8,468; item 14 not included) |
| total | | **about 177,000** |

Other notes:
- The MCC 128 bought from the US instead (about 30,000 + shipping) would reduce (a) by about 14,000.
- If the spacer kit is replaced by a few M2.5 standoffs, (a) drops by about 3,500-4,000.
- Shipping is not included. Digi-Key, Akizuki and Switch Science each charge their own shipping below their free-shipping thresholds.
