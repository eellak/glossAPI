# Greek page-routing dictionary

`el_GR.dic.gz` is the unmodified UTF-8 Hunspell el_GR 0.9 dictionary,
compressed with `gzip -n` (no timestamp or filename). It is loaded offline only
by routing profile v4 on image-dominant pages. This scope preserves the exact
unconfigured non-image routing of v3. Explicit word sets/paths and `GLOSSAPI_GREEK_WORDLIST`
take precedence and retain their historical all-page scope. V2/v3 do not acquire
an implicit dictionary. Reports record the effective dictionary scope.

Source: https://github.com/wooorm/dictionaries/blob/8cfea406b505e4d7df52d5a19bce525df98c54ab/dictionaries/el/index.dic

Upstream: https://github.com/stevestavropoulos/elspell

Source SHA256: `f08daefb302600beb1b345e4fd77f4ecf6617aa080a72efe6ae7eec0ad5b2ac7`.
Source size: 19,421,967 bytes; 828,806 entries plus a count header.
The loader verifies that hash and entry count, removes Hunspell `/flags`, then
normalizes NFC and case. It does **not** expand affixes, perform suggestions,
or measure recognition accuracy. Original bytes and the upstream licence notice
are included so the source is reproducible and redistributable.

Licence: MPL-1.1 OR GPL-2.0-or-later OR LGPL-2.1-or-later, as stated in
`LICENSE.el_GR`. GlossAPI distributes this independent data file under the
MPL-1.1 option; the notice and MPL text are included. The package's EUPL licence
does not replace the dictionary licence.

This is a modern Greek vocabulary. Valid polytonic, historical or specialist
text may have low/unknown validity and therefore incur extra OCR. Clean Latin-majority
image-dominant layers with sufficient text retain v3 routing and record
`validity_not_applicable`; see the page-routing contract in the migration guide.
Accent folding is still opt-in; a word hit is evidence of vocabulary membership,
not proof that an existing OCR layer matches the scan. An explicit empty word
set disables v4 validity measurement and exercises its unknown-to-OCR safety net.
The existing scan gate for fewer than 50 characters is retained even with a
positive word rate: a short caption cannot establish completeness of the body.

## English validity (experimental routing v5)

`en_US.dic.gz` is the unmodified UTF-8 SCOWL en_US 2020.12.07 Hunspell
source (`index.dic`) from wooorm/dictionaries revision
`8cfea406b505e4d7df52d5a19bce525df98c54ab`, compressed with gzip mtime 0.
Uncompressed SHA256: `f0b1a234bd178bdd01875b2a392a9647f888b8fe879f79c52aae62c2759b3647`.
Source: https://raw.githubusercontent.com/wooorm/dictionaries/8cfea406b505e4d7df52d5a19bce525df98c54ab/dictionaries/en/index.dic

Licence: MIT AND BSD (SCOWL's component notices and permissions are reproduced
in full in `LICENSE.en_US`). Lookup discards the count and Hunspell flags,
normalizes NFC and casefolds. There is no affix expansion or text correction.
The runtime verifies the source digest and entry count before using it.
