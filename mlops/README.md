MLOps libraries for Kubernetes cluster
====

## Getting Started

### Requirements

- Kubernetes cluster v1.12
- Helm v2.13.0  
  Follow instructions from https://github.com/helm/helm#install
- argo-events v0.7  
  Follow instructions from https://github.com/argoproj/argo-events/blob/v0.7/docs/quickstart.md

### Install batch

    $ helm install charts/batch --namespace argo-events

## Document

### charts

Helm charts for MLOps

#### batch

Batch execution chart
