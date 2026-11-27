# Publication ownership and distribution policy

**Status:** maintained internal contributor policy
**Last reviewed:** 2026-11-27

## The blog is the maintained edition

The personal blog is the authoritative, maintained home for substantial original articles. New long-form work is published there first, with a stable canonical URL. Corrections and technical updates are made on the blog first. A validated migrated or adapted article is published directly as a live blog post; it is not left as a Jekyll draft or an unpublished Studio record.

An article can have a different historical origin. Keep the original publication date and author accurate when material first appeared elsewhere. Keep historical source mappings in the private local registry; do not duplicate operational provenance in public front matter unless a technical or legal consumer requires it. Do not add migration banners or self-source links to the article. Remove phrases that narrate the author's own Medium, DEV, Virgool, Telegram, or Telegraph post, then write the supported technical point as standalone prose. Do not claim the blog was the historical first publication when it was not. For future work, publish on the blog first, then link platform copies back to the maintained blog edition.

Use public attribution for third-party research, required licenses, and other legitimate technical sources. Do not publicly attribute an article to the author's own secondary platform. A Telegram note expanded into an article is an adaptation; a Medium or DEV story imported with substantially equivalent content is an earlier edition. Keep Persian and English routes separate unless the articles are true translations. A distinct adaptation keeps its own canonical URL.

Canonical tags are signals, not commands. Search engines may choose a different canonical. A canonical tag does not establish that a page is indexed, and IndexNow acceptance is not indexing evidence. Do not use canonical metadata to point at materially different content.

## Distribution defaults

| Channel | Default use | Canonical / publication handling |
| --- | --- | --- |
| Personal blog | Complete, maintained article | Stable self-canonical; production checks remain the readiness evidence. |
| Medium | Selected equivalent full copy, or a distinct adaptation | Medium's official import tool sets the source URL as canonical. The author can set it in story settings. Verify the rendered tag on the external story. |
| DEV | Selected technical tutorial draft or equivalent copy | DEV supports `canonical_url` in front matter and in its API. Export with `published: false`; publish manually after reviewing the live blog URL. |
| Virgool | Persian adaptation with a distinct reader objective | Public pages inspected on 2026-11-27 declared their own canonical. An author-controlled external canonical setting was not verified; link to the blog edition and keep adaptations distinct. |
| Telegraph | Existing long-form authored article | A public story URL does not include the author's edit token. No external canonical control was verified; use the saved author edit link if available, and add a blog link without claiming canonical support. |
| Telegram | Short note, authored summary, or article announcement | No canonical relationship. Preserve existing messages. Bot API posting requires channel access, and bot message editing applies to messages the bot can edit; use a prepared draft and publish manually. |
| LinkedIn | Professional technical summary linking to the article | The official Posts API can create organic posts with authorized member permission. It is a social post workflow, not a verified full-article canonical workflow. Keep publication manual. |
| Hashnode | Optional future full copy | Current product materials describe a headless GraphQL API as a Pro feature. No integration is enabled; avoid recurring cost and verify canonical behavior before use. |
| GitHub | Code, reproducibility files, and relevant project documentation | Use the appropriate repository as the artifact source. Add article links only where they explain that artifact; do not change another repository as a distribution shortcut. |
| Hugging Face | Model cards, dataset cards, benchmark artifacts, and demos | Use the relevant Hub repository and its Markdown card. It complements an article; it is not an article syndication target. |
| X | Short technical observation or link | No full-copy workflow is enabled. API access and terms are not part of this implementation. |

Capabilities change. The references below are the evidence for this review; check them again before enabling an external integration. `UNKNOWN` is preferable to assuming an API, canonical control, analytics field, or permission exists.

## Provenance and local distribution records

Historical source URLs, source publication dates, original writing dates, blog publication dates, latest substantive revision dates, date evidence, and relationships to blog editions are kept in the private local registry rather than public article front matter. The blog page emits its normal self-canonical through the existing SEO layout. Never infer a precise time from a date-only source. For a substantially equivalent edition, preserve the verified source publication date. For an expansion based on an older note, preserve the note date as source evidence and use the actual blog publication date for the newly written edition. Set `last_modified_at` only when substantive content changes.

External URLs, historical source mappings, publication states, canonical observations, and revision fingerprints belong in the machine-local JSON registry under the user's local application-data directory. The repository script can validate and merge this file; it does not upload or publish registry data. Export or back up the file deliberately. The registry is local, not encrypted, and not a substitute for backup.

An external URL entered by itself does not mean the article was published or that its canonical was checked. Keep publication state and canonical state separate. Record a canonical as verified only after inspecting the external page's rendered `rel="canonical"` value. This shows the declared value, not the canonical later selected by a search engine.

## Full-copy readiness and revisions

Exports are preparation packages, not publication. Imported and adapted articles belong in the live blog with their supported historical date; do not leave them as Jekyll drafts. Before linking an external post to a blog article, confirm the target is deployed, returns HTTP 200, has its expected self-canonical and indexability, and is present in the sitemap. Use the site's existing production checks rather than adding another crawler to Studio. If production verification is unknown, do not prepare a link hook that directs readers to that URL.

Prefer full copies only where a platform's canonical behavior and author-side controls are verified. Keep original citations, code, tables, image references, and technical content. Exports preserve the source body and warn or stop on Jekyll-only Liquid; they do not silently render or remove it. A DEV export may remain unpublished as an external handoff package; that does not change the requirement that the blog edition itself is live. Telegram, LinkedIn, and Virgool outputs are editorial templates, not generated summaries. No tool in this repository publishes to an external platform.

When blog content changes materially, mark a full external copy for review. Do not overwrite external articles automatically. Short summaries need an update only when the changed claim is present in the summary. Formatting-only changes do not create a synchronization alert.

## Links and analytics

The distribution helper can create campaign links for approved blog URLs. Tracking parameters are added only to the share link; never to canonicals, sitemap URLs, or language alternates. The current campaign format uses `utm_source`, `utm_medium`, and `utm_campaign`. GoatCounter documents `utm_campaign`/`campaign` and `utm_source` as campaign inputs. Its visit and referrer counts are not search clicks, impressions, indexing state, or AI citations.

## Privacy, rights, and external actions

The public repository may contain article content and verified third-party references, license notices, and legally required attribution. Do not place self-source migration banners or operational migration metadata in public article front matter. Never commit browser-local distribution records, unpublished exports, API keys, analytics, private source paths, or personal knowledge files. Do not place provider tokens in Studio, site source, or exports.

Keep published external articles and Telegram messages unless the owner separately requests a change. Creating export files or campaign links does not publish them. The owner reviews and edits external posts, adds the blog link and article-specific hook, confirms external URLs, and checks canonical tags. No paid service, third-party publishing service, or external API automation is enabled.

## Verified platform evidence (reviewed 2026-11-27)

- [Google canonicalization guidance](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls) says canonical hints may be combined, but Google can choose another URL; it recommends consistent canonicals and sitemap signals.
- [Medium import instructions](https://help.medium.com/hc/en-us/articles/214550207-Importing-a-post-to-Medium) say an imported story is backdated and automatically receives the source URL as canonical. [Medium canonical settings](https://help.medium.com/hc/en-us/articles/360033930293-Set-a-canonical-link) say only the story author can set it in the web editor.
- [Forem API v1](https://developers.forem.com/api/v1) documents Markdown article creation with `published: false`, `canonical_url`, and an authenticated API key. [DEV's editor guide](https://dev.to/p/editor_guide) also documents canonical front matter and a four-tag limit.
- [Telegram Bot API](https://core.telegram.org/bots/api) documents `sendMessage` and `editMessageText`; [Telegram send errors](https://core.telegram.org/method/messages.sendMessage) include `CHAT_ADMIN_REQUIRED`. The Studio does not hold Telegram credentials.
- [LinkedIn Posts API](https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api?view=li-lms-2026-09) documents organic posts and `w_member_social`; access to other API permissions may require approval.
- [Hashnode's current product page](https://hashnode.com/) describes its headless GraphQL API under Pro, currently advertised at $5/month per seat. No paid feature is used here.
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards) and [dataset cards](https://huggingface.co/docs/hub/datasets-cards) are Markdown README artifacts with YAML metadata for their respective repositories.
- [GitHub repository contents API](https://docs.github.com/en/rest/repos/contents) supports authorized repository-file writes; it does not provide an article-canonical workflow.
- [GoatCounter campaign tracking](https://www.goatcounter.com/help/campaigns) recognizes `utm_campaign`/`campaign` and `utm_source` (among other source parameter names).
- **Observed, not officially documented:** on 2026-11-27 the public Medium profile showed 10 stories; the 3 stories mapped to active blog articles declared Medium self-canonicals. The public DEV profile showed 11 stories, all 11 declared DEV self-canonicals. The accessible Virgool author profile showed 4 authored stories; the 4 inspected public stories declared their own canonical (the public profile includes a not-found route on an unrelated guessed URL, excluded from the count). These checks do not reveal account-side settings or search-engine-selected canonicals.
