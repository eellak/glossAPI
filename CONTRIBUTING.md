# Contributing to GlossAPI

Thank you for your interest in contributing to GlossAPI! This document provides guidelines and instructions for contributing to this open-source project.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the expectations for all contributors.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improving GlossAPI, please first check if a similar issue already exists in the [issue tracker](https://github.com/eellak/glossAPI/issues). If not, feel free to create a new issue, providing as much detail as possible:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots or error messages if applicable
- Your environment (OS, Python version, etc.)

### Submitting Changes

1. **Fork the repository**
2. **Clone your fork locally**
   ```bash
   git clone https://github.com/YOUR-USERNAME/glossAPI.git
   cd glossAPI
   ```
3. **Create a new branch for your changes**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
   - Follow the coding style of the project
   - Add or update tests as necessary
   - Update documentation as needed
5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add a descriptive commit message"
   ```
6. **Push your changes to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Submit a pull request**
   - Go to the [original repository](https://github.com/eellak/glossAPI)
   - Click "New pull request"
   - Select "compare across forks"
   - Select your fork and branch
   - Add a clear description of your changes
   - Submit the pull request

## Development Guidelines

### Setting Up the Development Environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use meaningful variable and function names
- Write docstrings for all functions, classes, and modules
- Keep functions small and focused on a single responsibility

### Testing

- Write tests for all new functionality
- Run tests locally before submitting a pull request
- Ensure all tests pass

```bash
# Run tests
pytest
```

### Documentation

- Update documentation for any changes to the public API
- Use clear, concise language
- Include examples where appropriate

## Specific Guidelines for Downloader Scripts

When contributing new or improved downloader scripts:

1. **Respect website terms of service** - Only scrape from websites that allow it
2. **Implement polite scraping practices**:
   - Add appropriate delays between requests
   - Use a reasonable concurrency level
   - Identify your scraper with a user agent string
   - Handle errors gracefully
3. **Document your implementation**:
   - Create a README file explaining the approach
   - Document any issues encountered and how they were addressed
   - Include example commands
4. **Ensure robust error handling**:
   - Detect and handle rate limiting
   - Implement retries with backoff
   - Handle connection issues gracefully
5. **Follow the existing pattern**:
   - Customize the script for specific websites when needed
   - Keep compatibility with the overall GlossAPI architecture

## Pull Request Process

1. Update the README.md or other documentation with details of changes if appropriate
2. Update the version numbers in any examples files and the README.md to the new version
3. The PR will be merged once it has been reviewed and approved by a maintainer

## Project Structure

```
glossAPI/
├── pipeline/              # Data processing pipeline components
├── scraping/              # Web scraping and downloading scripts
├── Greek_variety_classification/ # Language variety classification
└── README.md             # Project documentation
```

## Contact

For questions or further assistance, please contact [glossapi.team@eellak.gr](mailto:glossapi.team@eellak.gr).

Thank you for contributing to GlossAPI!
