# Development Tasks

## Current Tasks

### High Priority
- [ ] Set up project structure and documentation
- [ ] Develop core metric learning architecture
- [ ] Develop core conditional flow matching architecture
- [ ] Set up experiment tracking with W&B

### Medium Priority
- [ ] Set up testing framework
- [ ] Implement evaluation metrics
- [ ] Create visualization utilities
- [ ] Set up model checkpointing

### Low Priority
- [ ] Add code quality tools
- [ ] Create example notebooks

## Completed Tasks
- [x] Create initial project structure
- [x] Set up basic documentation

## Task Details

### Set up project structure and documentation
**Status**: In Progress
**Priority**: High
**Dependencies**: None
**Requirements**:
- Create directory structure
- Set up documentation templates

### Develop core metric learning architecture
**Status**: In Progress
**Priority**: High
**Dependencies**: None
**Requirements**:
- Adapt code from NLOT and MFM metric learning approaches to our custom method
- Adapt these approaches to conduct _conditional_ metric learning, for different _conditional_ distribution manifolds

### Develop core conditional flow matching architecture
**Status**: In Progress
**Priority**: High
**Dependencies**: None
**Requirements**:
- Adapt code from MFM to allow conditions as inputs to flow matching velocity fields, allowing guided flow matching
- Adapt MFM flow matching code to incorporate our custom learned metric, which is adapted from their metric with updates from NLOT

### Set up experiment tracking
**Status**: Not Started
**Priority**: High
**Dependencies**: Development environment
**Requirements**:
- Initialize W&B project
- Set up logging utilities
- Create experiment configuration templates
- Implement metric tracking

## Requirements Tracking

### Core Requirements
1. Reproducible experiments
2. Comprehensive documentation
3. Efficient data processing
4. Robust model training
5. Clear evaluation metrics

### Documentation Requirements
1. Architecture documentation
2. Experiment documentation
3. Model documentation
4. Task tracking