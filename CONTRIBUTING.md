# Contributing to SAM Water Body Segmentation Project

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, GPU/CPU)
- Sample images or data if possible (ensure you have rights to share)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Use cases and benefits
- Any implementation ideas you might have

### Pull Requests

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, commented code
   - Follow the existing code style
   - Add docstrings to functions and classes
   - Update documentation if needed

3. **Test your changes**:
   - Ensure your code runs without errors
   - Test with different input scenarios
   - Verify outputs are correct

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: clear description of your changes"
   ```
   
   Use conventional commit messages:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** with:
   - A clear title and description
   - Reference to any related issues
   - Screenshots or examples if applicable

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/SAM_project.git
   cd SAM_project
   ```

2. Create a virtual environment:
   ```bash
   conda create -n sam_dev python=3.9
   conda activate sam_dev
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the SAM model checkpoint if needed.

## Code Style Guidelines

- Use **4 spaces** for indentation (not tabs)
- Follow **PEP 8** style guide for Python
- Keep functions focused and concise
- Add comments for complex logic
- Use meaningful variable names
- Maximum line length: 100-120 characters

## Testing

Before submitting a PR:
- Test with sample data from `data_set/dset-s2/tra_scene/`
- Verify outputs are generated correctly
- Check for any error messages or warnings
- Test with both CPU and GPU if possible

## Areas for Contribution

We especially welcome contributions in these areas:

1. **Performance Optimization**
   - Faster processing algorithms
   - Better memory management
   - GPU optimization

2. **New Features**
   - Support for additional satellite sensors
   - New spectral indices
   - Advanced post-processing techniques
   - GUI or web interface

3. **Documentation**
   - Tutorials and examples
   - API documentation
   - Use case studies

4. **Testing**
   - Unit tests
   - Integration tests
   - Test datasets

5. **Quality Improvements**
   - Code refactoring
   - Bug fixes
   - Error handling

## Questions?

If you have questions about contributing, feel free to:
- Open an issue labeled "question"
- Reach out to the maintainers

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🌊🛰️
