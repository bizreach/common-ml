{{- define "aws.templates" }}
- name: aws-initialize
  container:
    image: odaniait/aws-kubectl:latest
    command: [sh, -c]
    args:
      - |-
        # TODO: regionの取得処理統一
        export AWS_DEFAULT_REGION=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | tr -d '\n' | python -c 'import json; print(json.loads(input())["region"])')

        echo "Starting worker nodes"
        aws autoscaling set-desired-capacity --auto-scaling-group-name {{ .Release.Name }}-as --desired-capacity 1

        echo "Waiting join..."
        node=
        until [ "${node}" != "" ] && kubectl get nodes | grep ${node}; do
          sleep 1
          instance_id=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names {{ .Release.Name }}-as --query 'AutoScalingGroups[0].Instances[0].InstanceId' --output text)
          [ "${instance_id}" = "None" ] && continue
          : ${node:=$(aws ec2 describe-instances --instance-ids ${instance_id} --query 'Reservations[0].Instances[0].PrivateDnsName' --output text)}
        done

        echo "${node} joined."
        kubectl label nodes ${node} node-role.kubernetes.io/{{ .Release.Name }}=
        kubectl taint nodes ${node} node-role.kubernetes.io/{{ .Release.Name }}=:NoSchedule

        echo "Waiting ${node} node Ready..."
        until kubectl get pods -n argo-events --no-headers=true --field-selector=spec.nodeName=${node} | grep Running; do
          sleep 1
        done

        echo "done"
  activeDeadlineSeconds: 600
  tolerations:
  - key: "node-role.kubernetes.io/master"
    effect: "NoSchedule"

- name: aws-exit-handler
  steps:
  - - name: aws-sleep
      template: aws-sleep
  - - name: aws-finalize
      template: aws-finalize
  - - name: alice-notify
      template: alice-notify
      when: "{{"{{"}}workflow.status{{"}}"}} != Succeeded"

# TODO: sleepではなく、fluentdのAPIか何かをkickできないか？
- name: aws-sleep
  container:
    image: alpine:latest
    command: [sh, -c]
    args: ["sleep {{ .Values.aws.wait | default 120 }}"]
  tolerations:
  - key: "node-role.kubernetes.io/{{ .Release.Name }}"
    effect: "NoSchedule"

- name: aws-finalize
  container:
    image: odaniait/aws-kubectl:latest
    command: [sh, -c]
    args:
      - |-
        # TODO: regionの取得処理統一
        export AWS_DEFAULT_REGION=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | tr -d '\n' | python -c 'import json; print(json.loads(input())["region"])')

        instance_id=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names {{ .Release.Name }}-as --query 'AutoScalingGroups[0].Instances[0].InstanceId' --output text)
        node=$(aws ec2 describe-instances --instance-ids ${instance_id} --query 'Reservations[0].Instances[0].PrivateDnsName' --output text)

        echo "Deleting worker node ${node}"
        kubectl drain ${node} --ignore-daemonsets
        kubectl delete nodes ${node}

        echo "Stopping worker node ${node}"
        aws autoscaling set-desired-capacity --auto-scaling-group-name {{ .Release.Name }}-as --desired-capacity 0
  tolerations:
  - key: "node-role.kubernetes.io/master"
    effect: "NoSchedule"

{{- include "alice.notify" . }}
{{- end }}

{{- define "aws.jobs" }}
apiVersion: batch/v1
kind: Job
metadata:
  name: create-{{ .Release.Name }}-worker-node
  annotations:
    "helm.sh/hook": "pre-install,pre-upgrade"
    "helm.sh/hook-delete-policy": "before-hook-creation"
    argocd.argoproj.io/hook: PreSync
    argocd.argoproj.io/hook-delete-policy: HookSucceeded
spec:
  template:
    spec:
      containers:
      - name: creator
        image: odaniait/aws-kubectl:latest
        command: [sh, -c]
        args:
          - |-
            # TODO: regionの取得処理統一
            export AWS_DEFAULT_REGION=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | tr -d '\n' | python -c 'import json; print(json.loads(input())["region"])')
            ENVIRONMENT_NAME=$(aws cloudformation describe-stacks --stack-name ALICE-Stack --query 'Stacks[0].Parameters[?ParameterKey==`EnvironmentName`].ParameterValue' --output text)
            S3_BUCKET_NAME=${ENVIRONMENT_NAME}

            # TODO: If argocd supports post-delete helm hook, then use it.
            if aws cloudformation describe-stacks --stack-name "{{ .Release.Name }}" 2>/dev/null; then
              echo "Start delete"
              aws cloudformation delete-stack --stack-name "{{ .Release.Name }}"
              echo "Wait CF"
              aws cloudformation wait stack-delete-complete --stack-name "{{ .Release.Name }}"
            fi

            echo "Start create"
            aws cloudformation create-stack \
              --stack-name "{{ .Release.Name }}" \
              --template-url "https://s3-us-west-2.amazonaws.com/${S3_BUCKET_NAME}/infrastructure/aws/cloudformation/k8s-worker-node.yaml" \
              --parameters "[
                {
                  \"ParameterKey\": \"EnvironmentName\",
                  \"ParameterValue\": \"${ENVIRONMENT_NAME}\"
                },
                {
                  \"ParameterKey\": \"InstanceType\",
                  \"ParameterValue\": \"{{ .Values.aws.instanceType | default "t3.small" }}\"
                },
                {
                  \"ParameterKey\": \"SpotPrice\",
                  \"ParameterValue\": \"{{ .Values.aws.spotPrice | default 0.008 }}\"
                },
                {
                  \"ParameterKey\": \"VolumeSize\",
                  \"ParameterValue\": \"{{ .Values.aws.volumeSize | default 8 }}\"
                }
              ]"
            echo "Wait CF"
            aws cloudformation wait stack-create-complete --stack-name "{{ .Release.Name }}"
      restartPolicy: OnFailure
      tolerations:
      - key: "node-role.kubernetes.io/master"
        effect: "NoSchedule"
  backoffLimit: 1
{{- end }}
