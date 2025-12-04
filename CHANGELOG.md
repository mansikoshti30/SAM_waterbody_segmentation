# Changelog

All notable changes to the SAM Water Body Segmentation project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Docker container support
- Web-based UI
- Support for additional satellite sensors
- Automated accuracy assessment
- Multi-temporal analysis capabilities

## [1.0.0] - 2025-12-04

### Added
- Initial release of SAM Water Body Segmentation project
- Five different processing strategies (test_sam_1 through test_sam_5)
- NDWI and MNDWI spectral index computation
- SAM-based water body refinement
- Cloud masking capability
- Batch processing for multiple satellite images
- GeoJSON export with area calculations
- RGB overlay visualization
- Morphological post-processing
- Configurable parameters for different scenarios

### Features
- **test_sam_1_overall.py**: Basic NDWI + SAM implementation
- **test_sam_2.py**: Enhanced spectral index support
- **test_sam_3_overall.py**: Advanced morphological refinement
- **test_sam_4.py**: Experimental optimizations
- **test_sam_5_overall.py**: Production-ready with cloud masking

### Documentation
- Comprehensive README with installation and usage instructions
- INSTALL.md for detailed installation steps
- USAGE.md with examples and tips
- CONTRIBUTING.md for contributor guidelines
- Example configuration file

### Infrastructure
- requirements.txt for dependency management
- .gitignore for Python and data files
- MIT License
- GitHub Actions workflow for CI/CD

## Version History

### Version Descriptions

**v1.x (test_sam_1)**: Basic implementation
- Simple NDWI-based water detection
- Direct SAM application on candidate regions
- Basic output generation

**v2.x (test_sam_2)**: Enhanced indices
- Support for MNDWI when SWIR band available
- Improved spectral index selection
- Better normalization

**v3.x (test_sam_3)**: Morphological refinement
- Advanced morphological operations
- Better noise removal
- Improved polygon generation

**v4.x (test_sam_4)**: Experimental
- Performance optimizations
- Alternative processing strategies
- Memory efficiency improvements

**v5.x (test_sam_5)**: Production-ready
- Cloud masking integration
- Robust error handling
- IoU-based mask validation
- Controlled SAM expansion
- Small island removal
- Comprehensive output options

---

## Notes

- This project uses SAM (Segment Anything Model) from Meta AI
- Designed for Sentinel-2 multi-spectral satellite imagery
- Requires SAM model checkpoint file (not included in repository)
