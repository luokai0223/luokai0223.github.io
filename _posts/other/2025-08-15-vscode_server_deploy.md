---
title: 支持所有插件的 vscode server 部署
categories: [开发工具]
description: 支持所有插件的 vscode server 部署
keywords: 
- vscode
- web ide
- 开发工具
date: 2025-08-15
draft: false
---
### 前言

vscode 是多平台的文本编辑器，配合插件使用很适合作为python和ts的轻量化ide，同时vscode还可以部署到服务器上，通过网页打开，实现网页端的ide。不过现在网上的教程普遍是部署code-server，部署后大量插件不支持，比如lsp、各种AI编程辅助插件，本篇文章内容为如何部署vscode server，在浏览器端使用的同时也能支持绝大部分原生插件

### 部署包下载

vscode 官方就有服务端的部署包，地址为：

https://update.code.visualstudio.com/commit:b1c0a14de1414fcdaa400695b4db1c0799bc3124/server-win32-x64-web/stable

https://update.code.visualstudio.com/commit:b1c0a14de1414fcdaa400695b4db1c0799bc3124/server-linux-x64-web/stable

commit为版本号，可以安装一个桌面版后，在帮助-关于里面复制得到

### 部署方式

我使用的部署命令是：./bin/code-server --server-data-dir ./data/server --user-data-dir ./data/user  --host 0.0.0.0 --port 8080 --without-connection-token --accept-server-license-terms

部署完成后就可以在浏览器中打开，可以发现和code-server不一样，与桌面版功能一致，支持绝大部分原生插件，除了几个ssh-remote插件，因为已经运行在服务端了

#### 注意事项

直接部署时在浏览器是用http地址打开的，部分功能或者插件会因为http的原因显示异常，已知的是markdown预览功能、continue插件，可以通过转发到localhost地址或者配置https解决
