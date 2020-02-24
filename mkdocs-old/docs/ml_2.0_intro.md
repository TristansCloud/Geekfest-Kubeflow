# Introduction and Setup

**Objective:**

  * Open Kubeflow and create a Jupyter Notebook server

## Connect to Kubeflow

**Step 1** navigate to `173.246.132.94` in your browser. You'll be greeted with the Kubeflow homepage. At the top of the page you will have the option to select the namespace you will work in. You'll be assigned your namespace at the start of the workshop.

![namespace](./images/namespace.png)

## Setup Jupyter Notebook Server

**Step 1** You should now be in the Kubeflow home menu. The left hand side is for navigation. Take some time to click on the different menus. When you're finished, click the link to ```Notebook Servers```.

![clicknserver](./images/clicknotebookserver.png)

**Step 2** Ensure you are in the correct namespace from the dropdown menu at the top of the page. Then click **+ NEW SERVER**.

![newsever](./images/newsever.png)

**Step 3** Give a name to your Notebook server. Under image select custom image, and enter `docker.io/tristankcloud/kale-notebook:latest` Specify `0.5 CPU` and `1Gi` of memory. These are only memory and cpu requests, the pod can use more resources if needed.

Under Workspace Volume, change the size to `5Gi`. While we will not do this today, if you wanted to attach GPU to your Notebook you would do so under the Extra Resources menu.

Click launch to start the server, this may take a minute to be ready. Wait for a green check mark to appear in the status column and click connect when the server is running. You may recieve and error, wait a few seconds then refresh the page.

**Step 4** Launch a new terminal.

When it opens, paste the following code and hit enter
```
git clone https://github.com/TristansCloud/Geekfest-Kubeflow.git
```
Give it a second to refresh, but you should now see a directory named `Geekfest-Kubeflow`. 

**Step 6** Exit out of your terminal. On the left sidebar, click the second option from the top, the dark circle with the light square. Click `SHUT DOWN` to shut down the terminal.

!!! Congratulations
    You have set up your Notebook Server!