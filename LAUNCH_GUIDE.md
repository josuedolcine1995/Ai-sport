# 🚀 LAUNCH GUIDE — Go from this repo to a live store taking money (total cost: $0)

Follow these steps in order. Realistic time: **one afternoon.**

## Step 1 — Create your seller account (15 min, free)

You need one place that takes the payment AND delivers the files automatically. Best $0 options:

| Platform | Cost | Why |
|---|---|---|
| **Lemon Squeezy** (recommended) | $0 upfront, 5% + 50¢ per sale | Handles checkout, file delivery, AND sales tax/VAT for you (they're the "merchant of record" — big deal legally) |
| Gumroad | $0 upfront, 10% per sale | Simplest setup, has a built-in discovery marketplace |
| Stripe Payment Links | $0 upfront, ~2.9% + 30¢ | Cheapest fees, but you handle file delivery and taxes yourself |

→ Go to **lemonsqueezy.com**, sign up, verify your identity, and connect your bank account (this is how you get paid — it's required by law, there's no way around it, but it's free).

## Step 2 — Create the 4 products (30 min)

In your seller dashboard, create each product and upload the files from the `products/` folder:

1. **AI Prompt Vault for Small Business** — $19 — upload `products/ai-prompt-vault/AI-Prompt-Vault.md` (tip: open it, print → Save as PDF, upload the PDF — looks more premium)
2. **Freelancer Client Kit** — $24 — zip the 4 files in `products/freelancer-client-kit/` and upload the zip
3. **Money Reset: Budget & Savings Tracker** — $12 — zip the 2 files in `products/budget-tracker/`
4. **The Everything Bundle** — $39 — upload everything

Each product gives you a **checkout link**. Copy all 4.

## Step 3 — Wire up the store (5 min)

Open `config.js` in this repo and paste each checkout link between the quotes:

```js
checkoutLinks: {
  promptVault: "https://yourstore.lemonsqueezy.com/checkout/...",
  ...
}
```

Commit and push (or just ask Claude to do it for you).

## Step 4 — Put the site live, free (10 min)

Your store is a static site — GitHub Pages hosts it free with HTTPS:

1. On GitHub: your repo → **Settings → Pages**
2. Source: **Deploy from a branch** → pick your branch → folder: `/ (root)` → Save
3. In ~2 minutes your store is live at `https://YOURUSERNAME.github.io/REPONAME/`

(Later, a custom domain like `launchstack.shop` costs ~$3-10/year — it's the ONE thing worth paying for once you've made your first sales, because it doubles trust.)

## Step 5 — Test the whole pipeline (10 min)

- Open your live URL on your phone
- Click every buy button — does checkout open?
- In Lemon Squeezy, enable **Test Mode** and do a fake purchase — did the download email arrive?
- Turn off test mode. **You are now in business.**

## Step 6 — Get your first sale

Read `MARKETING_PLAYBOOK.md`. Your first sale will come from effort, not luck — the playbook is all free-traffic moves.

---

## Honest expectations (read this)

- A store with zero marketing makes zero dollars. The product is built; **traffic is now the whole game.**
- Realistic early numbers: ~1-3% of visitors buy. 100 visitors ≈ 1-3 sales ≈ $19-72. To make $500/month you need roughly 1,000-1,500 visitors/month. The playbook shows how to get them free.
- First sales usually take 1-3 weeks of consistent posting. Most people quit at day 10. Don't.
- Money from sales lands in your bank in 2-7 days depending on platform.
- Keep ~25-30% of profit aside for income taxes. When revenue gets real, talk to a tax person.
