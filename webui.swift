import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import WebKit

private final class JSONLineDecoder {
    private var buffer = Data()
    private let queue: DispatchQueue

    init(label: String) {
        queue = DispatchQueue(label: label)
    }

    func append(_ data: Data, onObject: @escaping ([String: Any]) -> Void) {
        queue.async { [weak self] in
            guard let self else { return }
            self.buffer.append(data)
            let newline = Data([0x0A])
            while let range = self.buffer.range(of: newline) {
                let line = self.buffer.subdata(in: self.buffer.startIndex..<range.lowerBound)
                self.buffer.removeSubrange(self.buffer.startIndex...range.lowerBound)
                guard !line.isEmpty,
                      let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
                onObject(object)
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKScriptMessageHandler, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var selectedInput: URL?
    private var worker: Process?
    private var workerInput: FileHandle?
    private var workerOutput: FileHandle?
    private var workerGeneration: UUID?
    private var isRunning = false
    private var isSelectingInput = false
    private var isTerminating = false

    private let rootURL: URL
    private var outputRootURL: URL { rootURL.appendingPathComponent("out", isDirectory: true) }
    private var webRootURL: URL { rootURL.appendingPathComponent("web", isDirectory: true) }

    override init() {
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: "--root"), arguments.indices.contains(index + 1) {
            rootURL = URL(fileURLWithPath: arguments[index + 1], isDirectory: true).standardizedFileURL
        } else {
            rootURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true).standardizedFileURL
        }
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)

        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.userContentController.add(self, name: "desynth")

        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1180, height: 820),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "DeSynth Local Studio"
        window.backgroundColor = NSColor(red: 0.067, green: 0.071, blue: 0.063, alpha: 1)
        window.minSize = NSSize(width: 780, height: 650)
        window.contentView = webView
        window.center()
        window.makeKeyAndOrderFront(nil)

        let indexURL = rootURL.appendingPathComponent("web/index.html")
        webView.loadFileURL(indexURL, allowingReadAccessTo: indexURL.deletingLastPathComponent())
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        isTerminating = true
        stopWorker()
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: "desynth")
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "desynth",
              message.frameInfo.isMainFrame,
              let body = message.body as? [String: Any],
              let action = body["action"] as? String else { return }

        switch action {
        case "ready":
            sendEnvironment()
        case "selectInput":
            chooseInput()
        case "run":
            guard let settings = body["settings"] as? [String: Any] else {
                sendToPage(["type": "error", "message": "设置格式无效"])
                return
            }
            run(settings: settings)
        case "revealOutput":
            if let path = body["path"] as? String, let url = validatedOutput(path: path) {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
        case "openOutput":
            if let path = body["path"] as? String, let url = validatedOutput(path: path) {
                NSWorkspace.shared.open(url)
            }
        case "quit":
            NSApp.terminate(nil)
        default:
            break
        }
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let url = navigationAction.request.url else {
            decisionHandler(.cancel)
            return
        }
        if url.isFileURL {
            let allowedPath = webRootURL.appendingPathComponent("index.html").standardizedFileURL.path
            decisionHandler(url.standardizedFileURL.path == allowedPath ? .allow : .cancel)
        } else if navigationAction.navigationType == .linkActivated,
                  ["http", "https"].contains(url.scheme?.lowercased() ?? "") {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
        } else {
            decisionHandler(.cancel)
        }
    }

    private func sendEnvironment() {
        let checks: [[String: Any]] = [
            check(label: "Python 环境", path: ".venv/bin/python"),
            checkAny(label: "GGUF 模型", paths: [
                "qwen-image-2512-Q4_K_M.gguf",
                "Qwen-Image-2512-Q4_K_M.gguf",
            ]),
            check(label: "Lightning LoRA", path: "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"),
            check(label: "Prompt Embeddings", path: "embeds_cache.pt"),
        ]
        let ready = checks.allSatisfy { ($0["ok"] as? Bool) == true }
        sendToPage([
            "type": "environment",
            "ready": ready,
            "checks": checks,
            "message": ready ? "" : "运行环境不完整，请先按 README 完成安装并下载模型。",
        ])
    }

    private func check(label: String, path: String) -> [String: Any] {
        let exists = FileManager.default.fileExists(atPath: rootURL.appendingPathComponent(path).path)
        return ["label": label, "ok": exists]
    }

    private func checkAny(label: String, paths: [String]) -> [String: Any] {
        let exists = paths.contains {
            FileManager.default.fileExists(atPath: rootURL.appendingPathComponent($0).path)
        }
        return ["label": label, "ok": exists]
    }

    private func chooseInput() {
        guard !isRunning, !isSelectingInput else { return }
        isSelectingInput = true
        let panel = NSOpenPanel()
        panel.title = "选择输入图像"
        panel.prompt = "选择"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"].compactMap {
            UTType(filenameExtension: $0)
        }
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let url = panel.url else {
                self?.isSelectingInput = false
                return
            }
            self?.setSelectedInput(url)
        }
    }

    private func setSelectedInput(_ url: URL) {
        let normalizedURL = url.standardizedFileURL
        let fileSize = (try? normalizedURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard fileSize > 0, fileSize <= 100 * 1024 * 1024 else {
            isSelectingInput = false
            sendToPage(["type": "notice", "message": "图片为空或超过 100 MB，请选择较小的文件。"])
            return
        }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let preview = self?.previewInfo(for: normalizedURL)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.isSelectingInput = false
                guard let preview else {
                    self.sendToPage(["type": "notice", "message": "无法读取这张图片，请换用 PNG、JPEG、WebP 或 TIFF。"])
                    return
                }
                self.selectedInput = normalizedURL
                var payload = preview
                payload["type"] = "inputSelected"
                payload["name"] = normalizedURL.lastPathComponent
                payload["path"] = normalizedURL.path
                self.sendToPage(payload)
            }
        }
    }

    private func run(settings: [String: Any]) {
        guard !isRunning else {
            sendToPage(["type": "notice", "message": "已有任务正在运行。"])
            return
        }
        guard let inputURL = selectedInput else {
            sendToPage(["type": "error", "message": "请先选择输入图片。"])
            return
        }
        do {
            try ensureWorker()
            isRunning = true
            try writeToWorker([
                "action": "run",
                "input": inputURL.path,
                "settings": settings,
            ])
        } catch {
            isRunning = false
            sendToPage(["type": "error", "message": error.localizedDescription])
        }
    }

    private func ensureWorker() throws {
        if let worker, worker.isRunning { return }

        let pythonURL = rootURL.appendingPathComponent(".venv/bin/python")
        let workerURL = rootURL.appendingPathComponent("webui_worker.py")
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            throw NSError(domain: "DeSynthWebUI", code: 1, userInfo: [NSLocalizedDescriptionKey: "找不到 .venv/bin/python，请先完成项目安装。"])
        }
        guard FileManager.default.fileExists(atPath: workerURL.path) else {
            throw NSError(domain: "DeSynthWebUI", code: 2, userInfo: [NSLocalizedDescriptionKey: "找不到 webui_worker.py。"])
        }

        let process = Process()
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let outputHandle = outputPipe.fileHandleForReading
        let generation = UUID()
        let decoder = JSONLineDecoder(label: "dev.desynth.webui.worker-output.\(generation.uuidString)")
        process.executableURL = pythonURL
        process.arguments = [workerURL.path]
        process.currentDirectoryURL = rootURL
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = FileHandle.standardError

        outputHandle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            decoder.append(data) { event in
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.workerGeneration == generation else { return }
                    self.handleWorkerEvent(event)
                }
            }
        }
        process.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                guard let self,
                      self.workerGeneration == generation,
                      self.worker === process else { return }
                self.workerOutput?.readabilityHandler = nil
                self.worker = nil
                self.workerInput = nil
                self.workerOutput = nil
                self.workerGeneration = nil
                if self.isRunning && !self.isTerminating {
                    self.isRunning = false
                    self.sendToPage([
                        "type": "error",
                        "message": "本机处理进程已意外退出（状态码 \(process.terminationStatus)）。",
                    ])
                }
            }
        }

        worker = process
        workerInput = inputPipe.fileHandleForWriting
        workerOutput = outputHandle
        workerGeneration = generation
        do {
            try process.run()
        } catch {
            outputHandle.readabilityHandler = nil
            worker = nil
            workerInput = nil
            workerOutput = nil
            workerGeneration = nil
            throw error
        }
    }

    private func writeToWorker(_ payload: [String: Any]) throws {
        guard let workerInput else {
            throw NSError(domain: "DeSynthWebUI", code: 3, userInfo: [NSLocalizedDescriptionKey: "本机处理进程尚未启动。"])
        }
        var data = try JSONSerialization.data(withJSONObject: payload)
        data.append(0x0A)
        try workerInput.write(contentsOf: data)
    }

    private func handleWorkerEvent(_ event: [String: Any]) {
        guard let type = event["type"] as? String else { return }
        if type == "complete" {
            isRunning = false
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                var enriched = event
                if let outputs = event["outputs"] as? [[String: Any]] {
                    enriched["outputs"] = outputs.map { output -> [String: Any] in
                        var item = output
                        if let path = output["path"] as? String,
                           let url = self.validatedOutput(path: path),
                           let preview = self.previewInfo(for: url) {
                            preview.forEach { item[$0.key] = $0.value }
                        }
                        return item
                    }
                }
                DispatchQueue.main.async { [weak self] in self?.sendToPage(enriched) }
            }
        } else {
            if type == "error" { isRunning = false }
            sendToPage(event)
        }
    }

    private func previewInfo(for url: URL) -> [String: Any]? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let width = (properties?[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue ?? 0
        let height = (properties?[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue ?? 0
        guard width > 0, height > 0, Int64(width) * Int64(height) <= 64_000_000 else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1600,
        ]
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary),
              let png = NSBitmapImageRep(cgImage: thumbnail).representation(using: .png, properties: [:]) else { return nil }
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return [
            "preview": "data:image/png;base64,\(png.base64EncodedString())",
            "width": width,
            "height": height,
            "size": size,
        ]
    }

    private func validatedOutput(path: String) -> URL? {
        let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        let resolvedRoot = outputRootURL.standardizedFileURL.resolvingSymlinksInPath()
        let rootPath = resolvedRoot.path.hasSuffix("/") ? resolvedRoot.path : resolvedRoot.path + "/"
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
        guard url.path.hasPrefix(rootPath),
              url.pathExtension.lowercased() == "png",
              values?.isRegularFile == true else { return nil }
        return url
    }

    private func sendToPage(_ payload: [String: Any]) {
        guard JSONSerialization.isValidJSONObject(payload) else { return }
        webView.callAsyncJavaScript(
            "window.desynthApp && window.desynthApp.handleNativeEvent(event)",
            arguments: ["event": payload],
            in: nil,
            in: .page
        ) { _ in }
    }

    private func stopWorker() {
        guard let process = worker else { return }
        workerOutput?.readabilityHandler = nil
        workerInput?.closeFile()
        workerGeneration = nil
        worker = nil
        workerInput = nil
        workerOutput = nil
        if process.isRunning { process.terminate() }
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.run()
