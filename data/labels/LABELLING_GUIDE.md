# Labelling guide (blind)

You label short US political texts from July to October 2026: tweets, Reddit posts and comments, and news headlines. Context: Republicans hold the presidency (Trump), the House and the Senate; the midterm elections are on Nov 3, 2026. "@user" replaces usernames and "http" replaces links.

Read each text on its own and give two labels. Do not look anything up and do not open any other file.

## 1. direction: which party does the text favour?

- **D**: the text is favourable to the Democrats, or critical of Republicans, Trump, MAGA, or the Trump administration and its policies. Also D: blaming Trump or Republicans for a problem (prices, gas, the war, cuts).
- **R**: the text is favourable to Republicans, Trump, or the administration and its policies, or critical of the Democrats, liberals, or the left. Also R: blaming Democrats for a problem.
- **N**: neither. Neutral reporting, mixed or balanced, both sides criticised equally, unclear, off-topic, or a pure question.

Judge the stance the author expresses, not the topic. A neutral headline that only reports a poll ("Democrats lead generic ballot by 8") is **N**, unless the wording clearly takes a side. Sarcasm counts by what it means ("Great job, Trump, gas is $4.50" is D).

## 2. tone: the overall emotional tone

- **pos**: positive or hopeful.
- **neg**: negative, angry, fearful, or critical.
- **neu**: neutral or factual.

## Output

A CSV with columns `item_id,direction,tone` (one row per item, every item, same order as the sheet). Values: direction in {D, R, N}; tone in {pos, neg, neu}. No other columns, no comments.
