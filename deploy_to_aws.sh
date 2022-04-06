#!/bin/bash
 
echo 'Starting to Deploy...'
ssh ubuntu@ec2-54-74-194-122.eu-west-1.compute.amazonaws.com " 
        cd ~/deployedApp
        gunicorn -b 127.0.0.1:8080 wsgi &
        "
echo 'Deployment completed successfully'
