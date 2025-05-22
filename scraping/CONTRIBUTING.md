# Contributing to GlossAPI

## Welcome Contributors! 🎉

Thank you for your interest in contributing to GlossAPI! This project aims to create a comprehensive open-source corpus for the Greek language to support language model development.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Specialized Downloaders](#specialized-downloaders)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Required packages
pip install aiohttp aiofiles asyncio argparse
```

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
```bash
git clone https://github.com/YOUR_USERNAME/glossAPI.git
cd glossAPI
```

3. **Add upstream remote**:
```bash
git remote add upstream https://github.com/eellak/glossAPI.git
```

## Development Environment Setup

### Recommended IDE Setup
- **JetBrains PyCharm** or **VS Code** with Python extensions
- **WSL2** for Windows users

### Project Structure
```
glossAPI/
├── scraping/
│   └── download_and_extract_scripts/
│       ├── downloader_*.py          # Specialized downloaders
│       ├── monitor_*.py             # Progress monitoring
│       ├── download_all_*.sh        # Automation scripts
│       └── README_*.md              # Documentation
├── pipeline/                        # Core processing pipeline
└── docs/                           # Project documentation
```

## Contributing Process

### 1. Issue-First Development
- **Check existing issues** before starting work
- **Create an issue** for new features or bugs
- **Get approval** from maintainers before major changes

### 2. Branch Strategy
```bash
# Create feature branch
git checkout -b feature/downloader-improvements

# Keep your branch updated
git fetch upstream
git rebase upstream/master
```

### 3. Development Workflow
```bash
# Make your changes
# Test thoroughly
# Commit with descriptive messages

git add .
git commit -m "feat: add specialized downloader for Kodiko repository

- Implement moderate concurrency (2 simultaneous downloads)
- Add comprehensive browser simulation headers
- Include PDF validation and Greek character handling
- Achieve 86.95% success rate (23,086/26,552 files)
"
```

### 4. Pull Request Process
1. **Push to your fork**:
```bash
git push origin feature/downloader-improvements
```

2. **Create Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/logs if applicable
   - Performance metrics if relevant

3. **Code Review**:
   - Address reviewer feedback
   - Keep discussions constructive
   - Update documentation as needed

## Code Standards

### Python Code Style
- Follow **PEP 8** guidelines
- Use **type hints** where appropriate
- Include **docstrings** for all functions
- Maximum line length: **88 characters**

### Downloader Specifications
```python
# Required structure for new downloaders
async def download_pdf(index, metadata, pdf_url, semaphore, args, user_agent):
    """
    Download a single PDF file with proper error handling.
    
    Args:
        index (int): File index for naming
        metadata (str): Document metadata
        pdf_url (str): URL to download
        semaphore (asyncio.Semaphore): Concurrency control
        args (argparse.Namespace): Command line arguments
        user_agent (str): Browser user agent string
        
    Returns:
        tuple: (success_bool, metadata, filename)
    """
```

### Error Handling Standards
```python
# Always include comprehensive error handling
try:
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            # Success handling
        else:
            logging.error(f"HTTP {response.status} for {url}")
except aiohttp.ClientError as e:
    logging.error(f"Client error: {e}")
except asyncio.TimeoutError:
    logging.error(f"Timeout error for {url}")
```

## Specialized Downloaders

### Site-Specific Requirements

| Repository | Concurrency | Sleep Time | Special Features |
|------------|-------------|------------|------------------|
| **Kodiko** | 2 | 1-4s | Legal document headers |
| **Greek Language** | 1 | 3-5s | Greek character support |
| **Cyprus Exams** | 1 | 3s | PDF validation |
| **Panelladikes** | 1 | 3s | URL fixing, dual logging |
| **Kallipos** | 1 | 1-6s | Multiple HTTP methods |

### Adding New Downloaders

1. **Create specialized script**: `downloader_SITENAME.py`
2. **Add automation script**: `download_all_SITENAME.sh`
3. **Add monitoring script**: `monitor_SITENAME.py`
4. **Create documentation**: `README_SITENAME.md`
5. **Update main README.md** with new repository

### Testing New Downloaders
```bash
# Test with small batch first
python downloader_SITENAME.py \
  --json ../../scraping/json_sitemaps/SITENAME_pdf.json \
  --type pdf \
  --req get \
  --output ../../downloads/SITENAME \
  --batch 5 \
  --sleep 3
```

## Testing Guidelines

### Manual Testing
- **Start small**: Test with batch size 1-5
- **Monitor logs**: Check for errors and warnings
- **Validate downloads**: Ensure files are valid PDFs
- **Measure performance**: Track success rates and timing

### Automated Testing
```python
# Example test structure
def test_pdf_validation():
    """Test PDF signature validation."""
    valid_pdf = b'%PDF-1.4...'
    invalid_pdf = b'<html>...'
    
    assert is_pdf(valid_pdf) == True
    assert is_pdf(invalid_pdf) == False
```

## Documentation

### Required Documentation

1. **README_SITENAME.md** for each new downloader:
   - Site description and challenges
   - Technical specifications
   - Usage examples
   - Performance results

2. **Code comments**:
   - Explain complex logic
   - Document site-specific workarounds
   - Include performance notes

3. **Commit messages**:
   - Use conventional commits format
   - Include performance metrics
   - Reference issues

### Documentation Standards
```markdown
## Performance Results
- **Success Rate**: 86.95% (23,086/26,552 files)
- **Average Speed**: 2.3 files/minute
- **Error Rate**: 13.05% (mainly HTTP 404)
- **Total Size**: 15.2 GB downloaded
```

## Community Guidelines

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time discussion with team
- **Email**: Direct contact with project maintainers

### Code of Conduct
- Be respectful and professional
- Help other contributors learn
- Focus on constructive feedback
- Respect server resources when scraping

### Contribution Recognition
Contributors will be acknowledged in:
- Project README.md
- Release notes
- GitHub contributor statistics
- Academic papers (where applicable)

## Common Issues and Solutions

### Issue: HTTP 500 Errors
**Solution**: Reduce concurrency, increase delays, check headers

### Issue: PDF Validation Failures
**Solution**: Implement content signature checking
```python
async def is_pdf(first_bytes):
    return first_bytes.startswith(b'%PDF')
```

### Issue: Greek Character Encoding
**Solution**: Use proper UTF-8 handling
```python
filename = re.sub(r'[<>:"/\\|?*]', '_', title)
```

## Getting Help

- **Documentation**: Check existing README files
- **Issues**: Search existing issues before creating new ones
- **Discord**: Join our community channel
- **Mentorship**: Reach out to experienced contributors

## Recognition

Major contributors to the downloader collection:
- **Nikos Tsekos**: Original downloader.py implementation
- **Development Team**: Specialized downloader implementations
- **Community**: Testing, feedback, and improvements

---

**Happy Contributing!** 🚀

Your contributions help build better Greek language resources for everyone.

*For questions about this contributing guide, please open an issue or contact the maintainers.*