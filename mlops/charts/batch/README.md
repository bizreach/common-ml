batch
====

## Configuration

The following table lists the configurable parameters of the batch chart and their default values.

| Parameter              | Description                                                                                                                              | Default                                                                                              |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| env                    | Environment of infrastructure. `aws` or `local` is available.                                                                            | aws                                                                                                  |
| aws.instanceType       | EC2 instance type.                                                                                                                       | t3.small                                                                                             |
| aws.spotPrice          | Price of spot instance.                                                                                                                  | "0.008"                                                                                              |
| aws.volumeSize         | Volume size.                                                                                                                             | "8"                                                                                                  |
| aws.access_key         | AWS access key ID.                                                                                                                       | null                                                                                                 |
| aws.secret_access_key  | AWS secret access key.                                                                                                                   | null                                                                                                 |
| argocd                 | [Deprecation] Whether to create autoscaling group.                                                                                       | false                                                                                                |
| mode                   | Production mode. <br />`workflow`: One time batch.<br />`webhook`: Batch kicked from webhook<br />`calendar`: Batch kicked from schedule | workflow                                                                                             |
| webhook.port           | Webhook port inner cluster.                                                                                                              | "12000"                                                                                              |
| calendar.schedule      | Cron schedule. `schedule` takes precedence over `interval`.                                                                              | "*/30 * * * *"                                                                                       |
| calendar.interval      | Interval duration.                                                                                                                       | 30m                                                                                                  |
| notify.hookUrl         | Slack's incoming webhook URL.                                                                                                            | "https://example.com/bizreach/common-ml/slack/notify"                                                |
| notify.iconEmoji       | Notification user's icon.                                                                                                                | null                                                                                                 |
| containerSteps         | Argo workflow step.                                                                                                                      | *1                                                                                                   |
| containerTemplates     | Argo workflow templates.                                                                                                                 | *2                                                                                                   |
| commonEnvVar           | Common envvars for each steps.                                                                                                           | []                                                                                                   |
| envVar                 | Envvars definitions per each steps.                                                                                                      | *3                                                                                                   |
| activeDeadlineSeconds  | Deadline for each steps.                                                                                                                 | 600                                                                                                  |
| volumes                | Volume definitions.                                                                                                                      | []                                                                                                   |
| workflow_creation_date | [Deprecation] Workflow creation date.                                                                                                    | "{{workflow.creationTimestamp.Y}}/{{workflow.creationTimestamp.m}}/{{workflow.creationTimestamp.d}}" |

*1)

```yaml
containerSteps:
- - name: hello1            # hello1 is run before the following steps
    template: whalesay
    arguments:
      parameters:
      - name: message
        value: "hello1"
- - name: hello2a           # double dash => run after previous step
    template: whalesay
    arguments:
      parameters:
      - name: message
        value: "hello2a"
  - name: hello2b           # single dash => run in parallel with previous step
    template: whalesay
    arguments:
      parameters:
      - name: message
        value: "hello2b"
```

*2)

```yaml
containerTemplates:
- name: whalesay
  inputs:
    parameters:
    - name: message
  container:
    image: docker/whalesay
    command: [cowsay]
    args: ["{{inputs.parameters.message}}"]
```

*3)

```yaml
envVar:
- name: hello1
  env: []
- name: hello2a
  env: []
- name: hello2b
  env: []
```
