{{- define "io.aws.s3" }}
{{- range  $dataIO := .Values.dataIO }}
- name: {{ $dataIO.name }}
  tolerations:
  - key: "node-role.kubernetes.io/{{ $.Release.Name }}"
    effect: "NoSchedule"
  container:
    image: aliceproj/boto3:latest
    volumeMounts:
      - mountPath: "/mnt"
        name: {{ $dataIO.hostPath }}
    env:
      - name: WORKFLOW_CREATION_DATE
        value: "{{ $.Values.workflow_creation_date }}"
    securityContext:
      runAsUser: 0
    command: [python, -c]
    args:
      - |-
        import os
        import sys
        import pathlib
        from datetime import datetime, timedelta
        import boto3

        {{ if eq $.Values.env "local" }}
        os.environ['AWS_ACCESS_KEY_ID'] = "{{ $.Values.aws.access_key }}"
        os.environ['AWS_SECRET_ACCESS_KEY'] = "{{ $.Values.aws.secret_access_key }}"
        {{ end }}

        s3 = boto3.resource('s3')
        bucket = s3.Bucket("{{ $dataIO.bucket }}")

        # set target_date
        target_date = "{{ $dataIO.targetDate }}"
        workflow_creation_date = os.environ['WORKFLOW_CREATION_DATE']

        if target_date == 'today':
            target_date = datetime.strptime(workflow_creation_date, '%Y/%m/%d')
        elif target_date == 'yesterday':
            target_date = datetime.strptime(
                workflow_creation_date, '%Y/%m/%d') - timedelta(days=1)
        elif 'days ago' in target_date:
            target_date = datetime.strptime(
                workflow_creation_date, '%Y/%m/%d') - timedelta(days=int(target_date.split()[0]))
        else:
            target_date = datetime.strptime(target_date, '%Y/%m/%d')

        Y = target_date.strftime('%Y')
        M = target_date.strftime('%m')
        D = target_date.strftime('%d')

        # make IO dir
        input_dir = '/mnt/input'
        output_dir = '/mnt/output'

        if not pathlib.Path(input_dir).exists():
            pathlib.Path(input_dir).mkdir()

        if not pathlib.Path(output_dir).exists():
            pathlib.Path(output_dir).mkdir()

        # connection s3
        files = [
        {{- range $dataIO.values }}
        '{{ . }}',
        {{- end }}
        ]

        for file in files:
            if "{{ $dataIO.type }}" == "download":
                bucket.download_file(f'{{ $dataIO.key }}/{file}', f'{input_dir}/{file}')
                print(f'download: {{ $dataIO.key }}/{file} to {input_dir}/{file}')
            elif "{{ $dataIO.type }}" == "upload":
                bucket.upload_file(f'{output_dir}/{file}', f'{{ $dataIO.key }}/{file}')
                print(f'upload: {output_dir}/{file} to {{ $dataIO.key }}/{file}')

        print("done")
{{- end }}
{{- end }}
