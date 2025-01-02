from bs4 import BeautifulSoup
import os
import re

def get_navbar_and_styles():
    return """
    <link rel="stylesheet" href="https://unpkg.com/bulma@1.0.0/css/bulma.min.css" />
    <script defer src="https://use.fontawesome.com/releases/v5.0.10/js/all.js"></script>
    <style>
        body {
            background-color: #ffffff;
            color: #6d4e48;
            font-family: Arial, sans-serif;
        }
        .navbar {
            background-color: #ffffff;
            border-bottom: 1px solid #6d4e48c5;
            margin-bottom: 20px;
        }
        .navbar-brand {
            display: flex;
            justify-content: flex-start;
            gap: 38px;
            padding-left: 15px;
        }
        .navbar-item {
            color: #6d4e48be;
            text-align: center;
        }
        .navbar-item:hover {
            color: #e5a831;
            background-color: #ffffff;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .content {
            font-size: 18px;
            line-height: 1.7;
            width: 100%;
            color: #6d4e48;
        }
        /* Text formatting */
        .content strong, .content b {
            font-weight: 700;
            color: inherit;
        }
        .content em, .content i {
            font-style: italic;
        }
        .content a {
            color: #6d4e48;
            text-decoration: underline;
        }
        .content a:hover {
            color: #e5a831;
        }
        /* Headers */
        .content h1 {
            font-size: 2em;
            font-weight: 600;
            margin-top: 1em;
            margin-bottom: 0.5em;
            color: #6d4e48;
        }
        .content h2 {
            font-size: 1.5em;
            font-weight: 600;
            margin-top: 0.83em;
            margin-bottom: 0.5em;
            color: #6d4e48;
        }
        .content h3 {
            font-size: 1.17em;
            font-weight: 600;
            margin-top: 0.67em;
            margin-bottom: 0.5em;
            color: #6d4e48;
        }
        /* Lists */
        .content ul {
            list-style: disc outside;
            margin-left: 2em;
            margin-top: 1em;
            color: #6d4e48;
        }
        .content ol {
            list-style: decimal outside;
            margin-left: 2em;
            margin-top: 1em;
            color: #6d4e48;
        }
        .content li {
            margin-bottom: 0.5em;
            color: #6d4e48;
        }
        /* Content width */
        .content p, 
        .content ul, 
        .content ol,
        .content h1,
        .content h2,
        .content h3,
        .content h4,
        .content h5,
        .content h6 {
            width: 100%;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            color: #6d4e48;
        }
        /* Images */
        .content img {
            width: 800px !important;
            max-width: 100%;
            height: auto !important;
            display: block;
            margin: 1.5em auto;
            object-fit: contain;
        }
        /* Original newsletter classes */
        .c8 { font-weight: 600; }
        .c20 { color: #6d4e48; }
        .c17 { font-style: italic; }
        .c1 { color: #6d4e48; }
        .c2 { color: #6d4e48; }
        .c3 { text-decoration: underline; }
    </style>
    """

def format_newsletter_html(content_html):
    """Format the newsletter HTML with navbar and styling."""
    return f"""
<!DOCTYPE html>
<html lang="el">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Newsletter - ΕΕΛΛΑΚ</title>
    <link rel="shortcut icon" href="../images/fav_icon.png" type="image/x-icon">
    <link rel="stylesheet" href="https://unpkg.com/bulma@1.0.0/css/bulma.min.css" />
    <script defer src="https://use.fontawesome.com/releases/v5.0.10/js/all.js"></script>
    <style>
        body {{
            background-color: #ffffff;
            color: #6d4e48;
            font-family: BlinkMacSystemFont, -apple-system, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", "Helvetica", "Arial", sans-serif;
        }}
        /* Navbar specific styles */
        .navbar {{
            background-color: #ffffff;
            border-bottom: 1px solid #6d4e48c5;
            margin-bottom: 20px;
        }}
        .navbar-brand {{
            display: flex;
            justify-content: flex-start;
            gap: 38px;
            padding-left: 15px;
        }}
        .navbar-item {{
            color: #6d4e48be;
            text-align: center;
        }}
        .navbar-item:hover {{
            color: #e5a831;
            background-color: #ffffff;
        }}
        /* Content styles */
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .content {{
            font-size: 18px;
            line-height: 1.7;
            width: 100%;
            color: #6d4e48;
        }}
        /* Text formatting */
        .content strong, .content b {{
            font-weight: 700;
            color: inherit;
        }}
        .content em, .content i {{
            font-style: italic;
        }}
        /* Main content link style - no hover effect */
        .content a {{
            color: #6d4e48;
            text-decoration: underline;
        }}
        /* Headers */
        .content h1 {{
            font-size: 2em;
            font-weight: 600;
            margin-top: 1em;
            margin-bottom: 0.5em;
            color: #6d4e48;
        }}
        .content h2 {{
            font-size: 1.5em;
            font-weight: 600;
            margin-top: 0.83em;
            margin-bottom: 0.5em;
            color: #6d4e48;
        }}
        .content h3 {{
            font-size: 1.17em;
            font-weight: 600;
            margin-top: 0.67em;
            margin-bottom: 0.5em;
            color: #6d4e48;
        }}
        /* Lists */
        .content ul {{
            list-style: disc outside;
            margin-left: 2em;
            margin-top: 1em;
            color: #6d4e48;
        }}
        .content ol {{
            list-style: decimal outside;
            margin-left: 2em;
            margin-top: 1em;
            color: #6d4e48;
        }}
        .content li {{
            margin-bottom: 0.5em;
            color: #6d4e48;
        }}
        /* Content width */
        .content p, 
        .content ul, 
        .content ol,
        .content h1,
        .content h2,
        .content h3,
        .content h4,
        .content h5,
        .content h6 {{
            width: 100%;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            color: #6d4e48;
        }}
        /* Images */
        .content img {{
            width: 800px !important;
            max-width: 100%;
            height: auto !important;
            display: block;
            margin: 1.5em auto;
            object-fit: contain;
        }}
        /* Original newsletter classes */
        .c8 {{ font-weight: 600; }}
        .c20 {{ color: #6d4e48; }}
        .c17 {{ font-style: italic; }}
        .c1 {{ color: #6d4e48; }}
        .c2 {{ color: #6d4e48; }}
        .c3 {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <nav class="navbar" role="navigation" aria-label="main navigation">
        <div class="navbar-brand">
            <a class="navbar-item" href="index.html">
                Λήμμα
            </a>
            <a class="navbar-item" href="omada.html">
                Ομάδα
            </a>
            <a class="navbar-item" href="keimena.html">
                Κείμενα
            </a>
        </div>
    </nav>
    <div class="container">
        <div class="content">
            {content_html}
        </div>
    </div>
</body>
</html>
"""

def get_title_from_newsletter(content_soup):
    """Extract title from the newsletter content."""
    # Try to find first heading
    first_heading = content_soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    if first_heading:
        return first_heading.get_text().strip()
    # If no heading, try first paragraph
    first_para = content_soup.find('p')
    if first_para:
        text = first_para.get_text().strip()
        # Return first 100 characters if text is too long
        return text[:100] + '...' if len(text) > 100 else text
    return "Newsletter"

def get_first_paragraph(content_soup):
    """Extract first paragraph or some preview text."""
    first_para = content_soup.find('p')
    if first_para:
        text = first_para.get_text().strip()
        return text[:200] + '...' if len(text) > 200 else text
    return "Διαβάστε το newsletter για περισσότερες πληροφορίες."

def create_newsletter_card(newsletter_path, title, preview_text):
    """Create a card element for the newsletter."""
    base_name = os.path.splitext(os.path.basename(newsletter_path))[0]
    card_id = f"newsletter-{base_name}"
    formatted_name = f"{base_name}_formatted.html"
    
    return f"""
    <div class="column is-one-third" style="padding: 2rem" id="{card_id}">
        <div class="card">
            <div class="card-content">
                <h2 class="title is-4">{title}</h2>
                <div class="content">
                    <p>{preview_text}</p>
                    <a href="{formatted_name}" class="button is-link">Διαβάστε το Newsletter</a>
                </div>
            </div>
        </div>
    </div>
    """

def update_keimena_html(newsletter_path):
    """Update keimena.html with the newsletter card."""
    # Read the newsletter content
    with open(newsletter_path, 'r', encoding='utf-8') as file:
        newsletter_content = file.read()
    newsletter_soup = BeautifulSoup(newsletter_content, 'html.parser')
    
    # Get title and preview text
    title = get_title_from_newsletter(newsletter_soup)
    preview_text = get_first_paragraph(newsletter_soup)
    
    # Read keimena.html
    with open('keimena.html', 'r', encoding='utf-8') as file:
        keimena_content = file.read()
    keimena_soup = BeautifulSoup(keimena_content, 'html.parser')
    
    # Create the new card
    new_card = BeautifulSoup(create_newsletter_card(newsletter_path, title, preview_text), 'html.parser')
    
    # Find the columns container
    columns_div = keimena_soup.find('div', class_='columns')
    if not columns_div:
        print("Error: Could not find columns container in keimena.html")
        return
    
    # Check if newsletter card already exists
    card_id = f"newsletter-{os.path.splitext(os.path.basename(newsletter_path))[0]}"
    existing_card = keimena_soup.find('div', id=card_id)
    if existing_card:
        # Replace existing card
        existing_card.replace_with(new_card)
    else:
        # Add new card at the beginning
        columns_div.insert(0, new_card)
    
    # Write updated content back to keimena.html
    with open('keimena.html', 'w', encoding='utf-8') as file:
        file.write(str(keimena_soup))

def main():
    # Read the newsletter HTML
    with open('Newsletter2024.html', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse content and extract body
    soup = BeautifulSoup(content, 'html.parser')
    content = ''.join(str(tag) for tag in soup.body.contents if tag.name)
    
    # Format the newsletter
    formatted_html = format_newsletter_html(content)
    
    # Write the formatted newsletter
    with open('Newsletter2024_formatted.html', 'w', encoding='utf-8') as file:
        file.write(formatted_html)
    
    # Update keimena.html with card
    update_keimena_html('Newsletter2024.html')

if __name__ == '__main__':
    main()
