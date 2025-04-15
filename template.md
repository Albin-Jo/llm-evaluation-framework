# LLM Evaluation Framework: Session Context

## Project Overview
LLM Evaluation Framework Overview

The LLM Evaluation Framework is an internal platform designed to assess the performance of domain-specific
AI agents developed using Azure OpenAI. 
It offers a comprehensive suite of tools for creating evaluations, managing datasets, 
comparing agent performance, and generating shareable reports.

Built with a FastAPI backend and Angular frontend, the framework seamlessly
integrates with our existing Azure OpenAI agents, while remaining extensible 
to support other LLM providers in the future. The initial implementation will 
prioritize core functionality, with advanced capabilities planned for subsequent phases.

Our organization has deployed multiple domain-specific agents (e.g., HR, Travel) powered by 
Azure OpenAI, each tailored to a specific business function. However, we currently lack standardized 
tools to systematically evaluate these agents, surface areas for improvement, and share performance
insights across teams. This framework aims to bridge that gap.
Project Goals:
Develop a unified dashboard to monitor and evaluate agent performance
Integrate with multiple evaluation tools (e.g., RAGAS, DeepEval, OPIK)
Support side-by-side comparisons of evaluation results
Manage datasets and prompts for consistent, repeatable testing
Generate and distribute evaluation reports via email
Lay the groundwork for continuous improvement of our AI agents


## Technical Stack
- **Backend**: FastAPI, Python, SQLAlchemy
- **Frontend**: Angular
- **Database**: PostgreSQL
- **Deployment**: Docker

## Current Project Status
- **Current Phase**: Implement reports module 
- **Last Session Date**: One day ago
- **Last Session Focus**: We have Implemented dataset, prompt Module, evaluation module and agent module
- **Current Focus**: Implement reports module

## Components Status

| Component | Status      | Notes                                              |
|-----------|-------------|----------------------------------------------------|
| Project Structure | Complete    | Basic FastAPI and Angular dashboard setup finished |
| Authentication | Hold        | OIDC integration 80% complete                      |
| Dataset Management | Complete    | All CRUD operations implemented                    |
| Agent Integration |  Started |                          |
| Evaluation Engine | Completed   |                                                    |
| Reports Module | Not Started |                              |
| Frontend Dashboard | Started     | Basic layout complete, components in development   |

## Today's Session Goal
Implement reports module and related files like schema, models, service etc.

## Key Considerations
sample of api requirements:
POST /api/reports/: Create a new report
GET /api/reports/: List all reports
GET /api/reports/{report_id}: Get report details
PUT /api/reports/{report_id}: Update report information
DELETE /api/reports/{report_id}: Delete a report
POST /api/reports/{report_id}/send: Send report via email
GET /api/reports/templates: List report templates
POST /api/reports/templates: Create a report template
GET /api/reports/templates/{template_id}: Get template details


## Code References
- Backend Repository Structure: [Link to Gist if applicable]
- Frontend Components: [Link to Gist if applicable]
- API Specifications: [Link to Gist if applicable]

## Immediate Next Steps
1. [First specific task for today's session]
2. [Second specific task for today's session]
3. [Third specific task for today's session] 