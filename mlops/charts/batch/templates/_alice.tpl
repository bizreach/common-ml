{{- define "alice.template" }}
{{ $envVar := .Values.envVar }}
{{ $commonEnvVar := .Values.commonEnvVar }}
{{ $activeDeadlineSeconds := .Values.activeDeadlineSeconds }}
{{- range .Values.containerTemplates }}
  {{ $currentTemplate := . }}
  {{ $_ := set $currentTemplate.container "env" $commonEnvVar}}
  {{- range $envVar }}
    {{ if eq $currentTemplate.name .name }}
      {{- range .env }}
        {{ $_ := (append $currentTemplate.container.env . | set $currentTemplate.container "env") }}
      {{- end }}
    {{ end }}
  {{- end }}
- activeDeadlineSeconds: {{ $activeDeadlineSeconds }}
  tolerations:
  - key: "node-role.kubernetes.io/{{ $.Release.Name }}"
    effect: "NoSchedule"
{{ toYaml $currentTemplate | indent 2 }}
{{- end }}
{{- end }}

{{- define "alice.notify" }}
- name: alice-notify
  container:
    image: byrnedo/alpine-curl:latest
    command: [sh, -c]
    args:
      - |-
        MENTION="<!here>"
        WORKFLOW_NAME={{"{{"}}workflow.name{{"}}"}}
        TIMESTAMP="{{"{{"}}workflow.creationTimestamp{{"}}"}}"

        failed_data=`cat << EOF
          payload={
            "username": "alice notify",
            {{ if .Values.notify.iconEmoji }}"icon_emoji": {{ .Values.notify.iconEmoji | quote }},{{ end }}
            "text": "$MENTION",
            "attachments": [{
              "fallback": "Failed batch process",
              "color": "#E96D76",
              "title": "Failed",
              "text": "Workflow: $WORKFLOW_NAME \nTimestamp: $TIMESTAMP"
            }]
          }
        EOF`

        curl -X POST --data-urlencode "$failed_data" {{ .Values.notify.hookUrl | quote }}
  tolerations:
  - key: "node-role.kubernetes.io/master"
    effect: "NoSchedule"
{{- end }}
