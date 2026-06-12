#!/usr/bin/env bash
# Creates all 4 LaunchStack products + payment links in Stripe automatically.
#
# Usage:
#   STRIPE_KEY=rk_live_xxx SITE_URL=https://josuedolcine1995.github.io/Ai-sport ./scripts/create_stripe_products.sh
#
# Use a RESTRICTED key (Stripe dashboard → Developers → API keys → Create
# restricted key) with WRITE access to: Products, Prices (under "Products"),
# and Payment Links — nothing else. Delete the key after setup.
set -euo pipefail

: "${STRIPE_KEY:?Set STRIPE_KEY to your restricted Stripe API key}"
SITE_URL="${SITE_URL:-https://josuedolcine1995.github.io/Ai-sport}"
API="https://api.stripe.com/v1"

create() { # name, description, amount_cents, product_key
  local name="$1" desc="$2" amount="$3" pkey="$4"

  local product_id price_id link_url
  product_id=$(curl -sf "$API/products" -u "$STRIPE_KEY:" \
    -d "name=$name" -d "description=$desc" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

  price_id=$(curl -sf "$API/prices" -u "$STRIPE_KEY:" \
    -d "product=$product_id" -d "unit_amount=$amount" -d "currency=usd" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

  link_url=$(curl -sf "$API/payment_links" -u "$STRIPE_KEY:" \
    -d "line_items[0][price]=$price_id" -d "line_items[0][quantity]=1" \
    -d "after_completion[type]=redirect" \
    --data-urlencode "after_completion[redirect][url]=$SITE_URL/thanks.html?p=$pkey" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['url'])")

  echo "$pkey: $link_url"
}

create "AI Prompt Vault for Small Business" \
  "105 copy-paste ChatGPT and Claude prompts in 10 business categories. Instant download, lifetime updates." \
  1900 promptVault

create "Freelancer Client Kit" \
  "Contract, proposal, invoice, and client onboarding templates. Everything from yes to paid. Instant download." \
  2400 clientKit

create "Money Reset: Budget and Savings Tracker" \
  "Printable monthly budget, 52-week savings challenge, and debt snowball planner. Instant download." \
  1200 budgetTracker

create "The Everything Bundle" \
  "All three LaunchStack toolkits: AI Prompt Vault, Freelancer Client Kit, and Money Reset Tracker. \$55 of tools for \$39." \
  3900 bundle

echo
echo "Done. Paste the 4 URLs above into config.js (or let Claude do it)."
