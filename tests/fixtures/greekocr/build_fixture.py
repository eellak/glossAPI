"""Rebuild the synthetic image-only fixture; Times New Roman stays on this Mac."""
from pathlib import Path
import hashlib, json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
ROOT = Path(__file__).resolve().parent
FONT = Path('/System/Library/Fonts/Supplemental/Times New Roman.ttf')
PAGES = [
 ['Μῆνιν ἄειδε, θεά, Πηληϊάδεω Ἀχιλῆος',
  'οὐλομένην, ἣ μυρί᾽ Ἀχαιοῖς ἄλγε᾽ ἔθηκε,',
  'πολλὰς δ᾽ ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν'],
 ['ἡρώων, αὐτοὺς δὲ ἑλώρια τεῦχε κύνεσσιν',
  'οἰωνοῖσί τε πᾶσι, Διὸς δ᾽ ἐτελείετο βουλή,',
  'ἐξ οὗ δὴ τὰ πρῶτα διαστήτην ἐρίσαντε']]

def main():
 font = ImageFont.truetype(str(FONT), 58)
 images = []
 for lines in PAGES:
  im = Image.new('RGB', (1500, 2000), (250, 249, 246))
  d = ImageDraw.Draw(im)
  for i, text in enumerate(lines):
   assert d.textbbox((0,0), text, font=font)[2] <= 1260
   d.text((120, 530+i*115), text, font=font, fill=(29,27,25))
  images.append(im.filter(ImageFilter.GaussianBlur(0.35)))
 images[0].save(ROOT/'polytonic_scan.pdf', save_all=True, append_images=images[1:],
                resolution=300.0, quality=85, subsampling=0)
 (ROOT/'source.json').write_text(json.dumps(dict(
   source='Homer, Iliad 1.1-6, ancient Greek original, public domain; manually transcribed for this fixture',
   font='Times New Roman regular', font_sha256=hashlib.sha256(FONT.read_bytes()).hexdigest(),
   dpi=300, image_size=[1500,2000], page_size_points=[360,480],
   appearance='off-white paper, dark grey ink, 0.35px blur; no hidden text layer',
   pages=PAGES), ensure_ascii=False, indent=2)+'\n')
if __name__ == '__main__': main()
