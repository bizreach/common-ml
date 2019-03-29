{{- define "workflow" }}
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  {{ if eq .Values.mode "workflow" }}
  name: "{{ .Release.Name }}-workflow-{{ randAlphaNum 5 | lower }}" # cannot use generateName with Helm
  {{ else }}
  generateName: "{{ .Release.Name }}-workflow-"
  {{ end }}
spec:
  entrypoint: step-run
{{ if or (eq .Values.env "AWS") (eq .Values.env "aws") }}
  onExit: aws-exit-handler
{{ end }}
  templates:
  # workflow step definition.
  - name: step-run
    steps:
{{ if or (eq .Values.env "AWS") (eq .Values.env "aws") }}
    - - name: aws-initialize
        template: aws-initialize
{{ end }}
{{ toYaml .Values.containerSteps | indent 4 }}
  # container definitions.
{{- include "alice.template" . | indent 2 }}
{{ if or (eq .Values.env "AWS") (eq .Values.env "aws") }}
  # utility container definitions for AWS.
{{- include "aws.templates" . | indent 2 }}
{{ end }}
{{- include "io.aws.s3" . | indent 2 }}
  # volume definitions.
  volumes:
{{ toYaml .Values.volumes | indent 4 }}
{{- end }}
