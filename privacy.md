---
layout: publication
lang: en
title: "Privacy — Mohammad Shojaei"
description: "How this publication and its browser-based Publication Studio handle analytics and local data."
permalink: /privacy/
---

# Privacy

This publication uses GoatCounter to understand aggregate visits and referral sources. On public pages, the site loads GoatCounter's versioned script unless the browser has opted out. The request can include the page path, title, referrer, and screen-width information. GoatCounter's published privacy documentation describes its aggregation and short-lived request processing; see its [privacy policy](https://www.goatcounter.com/help/privacy) and [GDPR consent notes](https://www.goatcounter.com/help/gdpr). The Publication Studio, retired pages, and local previews are not tracked. No analytics events, advertising pixels, session replay, or cross-site profile are configured here.

The site's owner views traffic in the private [GoatCounter dashboard](https://mshojaei77.goatcounter.com/). This publication does not expose analytics API keys or private traffic reports. To opt out of GoatCounter in a browser, use its documented `#toggle-goatcounter` URL control; a browser or network tracker blocker may also block the request.

## Publication Studio

Publication Studio saves drafts and staged images in IndexedDB in the current browser profile. Signing in uses a GitHub token supplied by the publisher. The token stays in tab memory by default; if the publisher explicitly selects **Remember me**, Studio stores it in that browser's local storage for up to seven days. Same-origin scripts can access local storage, so the option is intended only for a trusted device and trusted browser profile. The published site does not receive draft data until the publisher submits a change to GitHub.

This notice describes the behavior implemented by this site. External services have their own policies, and this notice is not a legal compliance determination.
