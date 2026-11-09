from glossapi.gloss_downloader import GlossDownloader


def test_detects_waf_challenge_html(tmp_path):
    downloader = GlossDownloader(output_dir=str(tmp_path))
    url = "https://eur-lex.europa.eu/legal-content/EL/TXT/PDF/?uri=OJ:L_202502360"
    headers = {
        "Content-Type": "text/html; charset=UTF-8",
        "x-amzn-waf-action": "challenge",
    }
    body = b"""<!DOCTYPE html><html><body>
    <script>AwsWafIntegration.getToken()</script>
    <noscript>verify that you're not a robot</noscript>
    </body></html>"""

    assert downloader.infer_file_extension(url, headers, body) == "html"
    error = downloader._detect_html_interstitial(url, headers, body)

    assert error is not None
    assert "challenge page" in error.lower()


def test_detects_js_document_viewer_html(tmp_path):
    downloader = GlossDownloader(output_dir=str(tmp_path))
    url = "https://freader.ekt.gr/eadd/index.php?doc=60819&lang=el"
    headers = {
        "Content-Type": "text/html; charset=UTF-8",
    }
    body = b"""<!DOCTYPE html><html><head>
    <meta name="monitor-signature" content="monitor:player:html5">
    <script>bookConfig.totalPageCount = 236;</script>
    <script>var fliphtml5_pages = [{"l":"../getfile.php?lib=eadd&path=large&item=1.jpg"}];</script>
    <script src="javascript/LoadingJS.js"></script>
    </head></html>"""

    assert downloader.infer_file_extension(url, headers, body) == "html"
    error = downloader._detect_html_interstitial(url, headers, body)

    assert error is not None
    assert "document viewer" in error.lower()


def test_regular_html_document_is_still_allowed(tmp_path):
    downloader = GlossDownloader(output_dir=str(tmp_path))
    url = "https://example.org/article"
    headers = {
        "Content-Type": "text/html; charset=UTF-8",
    }
    body = b"""<!DOCTYPE html><html><head><title>Article</title></head>
    <body><main><article><h1>Normal HTML document</h1><p>Body text.</p></article></main></body></html>"""

    assert downloader.infer_file_extension(url, headers, body) == "html"
    assert downloader._detect_html_interstitial(url, headers, body) is None
