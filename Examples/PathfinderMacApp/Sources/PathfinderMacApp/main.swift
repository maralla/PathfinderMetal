import AppKit
import Metal
import simd
import PathfinderMetal

final class MetalView: NSView {
    override var wantsUpdateLayer: Bool { true }

    private var device: MTLDevice! = MTLCreateSystemDefaultDevice()
    private let metalLayer = CAMetalLayer()
    private var canvas: Canvas!

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setup()
    }
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }
    private func setup() {
        wantsLayer = true
        guard let metal = device else { 
            print("Error: No Metal device available")
            return 
        }
        
        guard let baseLayer = self.layer else { 
            print("Error: No base layer available")
            return 
        }
        
        // Configure CAMetalLayer properly
        metalLayer.device = metal
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = true
        metalLayer.isOpaque = false
        
        // Set proper color space
        metalLayer.colorspace = CGColorSpace(name: CGColorSpace.displayP3)
        
        let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        
        metalLayer.frame = baseLayer.bounds
        let drawableSize = CGSize(
            width: bounds.width * scale,
            height: bounds.height * scale
        )
        metalLayer.drawableSize = drawableSize
        
        // Make sure our layer is visible and front-most
        baseLayer.addSublayer(metalLayer)
        
        let size = SIMD2<Float32>(Float32(drawableSize.width), Float32(drawableSize.height))
        
        canvas = Canvas(size: size)

        render()
    }
    
    // Helper to get next power of 2
    private func nextPowerOf2(_ n: Int32) -> Int32 {
        var p: Int32 = 1
        while p < n {
            p *= 2
        }
        return p
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        needsDisplay = true
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        needsDisplay = true
    }
    
    private func render() {
        guard let drawable = metalLayer.nextDrawable() else {
            return
        }
        
        canvas.draw { ctx in
            ctx.lineWidth = 2
            
            let path = CGMutablePath()
            path.move(to: .init(x: 520, y: 520))
            path.addCurve(to: .init(x: 720, y: 320), control1: .init(x: 560, y: 260), control2: .init(x: 680, y: 260))
            path.addCurve(to: .init(x: 520, y: 320), control1: .init(x: 560, y: 320), control2: .init(x: 540, y: 360))
            path.closeSubpath()

            ctx.stroke(path)
            ctx.stroke(addEllipse(360, 420, 90, 60))
            ctx.stroke(curve1())
            ctx.stroke(makeStar(cx: 620, cy: 620, rOuter: 70, rInner: 30, points: 5))
            
            var outer = makeEllipseOutline(cx: 860, cy: 250, rx: 70, ry: 48, reverse: false)
            var inner = makeEllipseOutline(cx: 860, cy: 250, rx: 35, ry: 24, reverse: true)

            let donut = CGMutablePath()
            donut.addPath(outer)
            donut.addPath(inner)
            ctx.fill(donut, rule: .evenOdd)
            
            let rose = makeRose(cx: 960, cy: 360, a: 70, k: 4, steps: 400)

            canvas.draw { ctx in
                ctx.fillStyle = .color(NSColor.systemMint.cgColor)
                ctx.fill(rose)
            }
            ctx.stroke(rose)
            
            var (hachurePath, border, hachureClip) = hachureRect(
                x: 120,
                y: 800,
                w: 240,
                h: 190,
                gap: 10,
                angle: .pi / 6,
                strokeWidth: 4
            )
            
            canvas.draw { ctx in
                ctx.mask(path: hachureClip)
                ctx.stroke(hachurePath)
            }
            
            ctx.stroke(border)
            
            let rect = CGMutablePath()
            rect.addRect(.init(origin: .init(x: 60, y: 70), size: .init(width: 240, height: 190)))
            
            canvas.draw { ctx in
                let origin = SIMD2<Float>(60, 70)
                let center = origin + SIMD2<Float>(240, 190) * 0.5

                ctx.transform = CGAffineTransform(translationX: CGFloat(center.x), y: CGFloat(center.y))
                    .rotated(by: .pi / 4)
                    .translatedBy(x: -CGFloat(center.x), y: -CGFloat(center.y))
                ctx.stroke(rect)
            }

            ctx.stroke(rect)

            canvas.draw { ctx in
                ctx.lineDash = [5, 6]
                
                let path = CGMutablePath()
                path.addRect(.init(origin: .init(x: 520, y: 800), size: .init(width: 300, height: 250)))
                ctx.stroke(path)
            }

            let dashRect2 = CGMutablePath()
            dashRect2.addRect(.init(origin: .init(x: 620, y: 800), size: .init(width: 300, height: 250)))
            ctx.stroke(dashRect2)

            canvas.draw { ctx in
                ctx.strokeStyle = .color(NSColor.systemRed.cgColor)
                ctx.lineJoin = .round
                ctx.lineWidth = 5

                let redRect = CGMutablePath()
                redRect.addRect(.init(origin: .init(x: 990, y: 800), size: .init(width: 300, height: 250)))
                ctx.stroke(redRect)
            }

            let blackRect = CGMutablePath()
            blackRect.addRect(.init(origin: .init(x: 990, y: 500), size: .init(width: 300, height: 250)))
            ctx.stroke(blackRect)

            canvas.draw { ctx in
                ctx.strokeStyle = .color(NSColor.systemGreen.cgColor)
                ctx.lineWidth = 5
                ctx.lineJoin = .round

                let line = CGMutablePath()
                line.move(to: .init(x: 1000, y: 200))
                line.addLine(to: .init(x: 1400, y: 300))
                line.addLine(to: .init(x: 1500, y: 800))

                ctx.stroke(line)
            }
            
            canvas.draw { ctx in
                ctx.lineWidth = 4
                ctx.fillStyle = .color(NSColor.systemIndigo.cgColor)
                ctx.lineJoin = .round
                
                let m = CGMutablePath()
                m.move(to: .init(x: 30, y: 400))
                m.addLine(to: .init(x: 90, y: 700))
                m.addLine(to: .init(x: 400, y: 550))
                m.closeSubpath()
                
                ctx.fill(m)
                ctx.stroke(m)
            }
        }
        
        let size = CGSize(
            width: bounds.width,
            height: bounds.height
        )

        
        canvas.render(
            on: drawable,
            device: .init(device: device),
            options: .init()
        )
    }
    
    func hachureRect(
        x: CGFloat,
        y: CGFloat,
        w: CGFloat,
        h: CGFloat,
        gap: CGFloat = 10,
        angle: CGFloat = .pi / 12,
        strokeWidth: CGFloat = 3
    ) -> (CGPath, CGPath, CGPath) {
        // Draw angled hatch lines first
        let diag = hypot(w, h) * 1.2
        let cosA = cos(angle)
        let sinA = sin(angle)
        let count = Int(ceil((w + h) / gap)) + 2

        let fillPath = CGMutablePath()
        for i in -1..<count {
            let offset = (CGFloat(i) * gap) - h
            let cx = x + w * 0.5
            let cy = y + h * 0.5
            let dx = cosA
            let dy = sinA
            let px = cx + offset * (-dy)
            let py = cy + offset * (dx)
            let x0 = px - dx * diag
            let y0 = py - dy * diag
            let x1 = px + dx * diag
            let y1 = py + dy * diag
            fillPath.move(to: .init(x: x0, y: y0))
            fillPath.addLine(to: .init(x: x1, y: y1))
        }

        // Draw border last so it's not covered by hatches
        let border = CGMutablePath()
        border.addRect(.init(origin: .init(x: x, y: y), size: .init(width: w, height: h)))

        var hachureClip = CGMutablePath()
        hachureClip.addRect(.init(origin: .init(x: x, y: y), size: .init(width: w, height: h)))

        return (fillPath, border, hachureClip)
    }
    
    func makeRose(cx: CGFloat, cy: CGFloat, a: CGFloat, k: CGFloat, steps: Int) -> CGPath {
        let p = CGMutablePath()
        for i in 0...steps {
            let t = CGFloat(i) / CGFloat(steps) * (.pi * 2)
            let r = a * cos(k * t)
            let x = cx + r * cos(t)
            let y = cy + r * sin(t)
            if i == 0 { p.move(to: .init(x: x, y: y)) } else { p.addLine(to: .init(x: x, y: y)) }
        }
        p.closeSubpath()
        return p
    }
    
    func makeEllipseOutline(
        cx: CGFloat,
        cy: CGFloat,
        rx: CGFloat,
        ry: CGFloat,
        reverse: Bool = false
    )
        -> CGPath
    {
        let p = CGMutablePath()
        let kappa: CGFloat = 0.5522847498307936
        let ox = rx * kappa
        let oy = ry * kappa
        p.move(to: CGPoint(x: cx - rx, y: cy))
        if !reverse {
            p.addCurve(
                to: CGPoint(x: cx, y: cy - ry),
                control1: CGPoint(x: cx - rx, y: cy - oy),
                control2: CGPoint(x: cx - ox, y: cy - ry)
            )
            p.addCurve(
                to: CGPoint(x: cx + rx, y: cy),
                control1: CGPoint(x: cx + ox, y: cy - ry),
                control2: CGPoint(x: cx + rx, y: cy - oy)
            )
            p.addCurve(
                to: CGPoint(x: cx, y: cy + ry),
                control1: CGPoint(x: cx + rx, y: cy + oy),
                control2: CGPoint(x: cx + ox, y: cy + ry)
            )
            p.addCurve(
                to: CGPoint(x: cx - rx, y: cy),
                control1: CGPoint(x: cx - ox, y: cy + ry),
                control2: CGPoint(x: cx - rx, y: cy + oy)
            )
        } else {
            // Trace the ellipse in the opposite winding direction
            p.addCurve(
                to: CGPoint(x: cx, y: cy + ry),
                control1: CGPoint(x: cx - rx, y: cy + oy),
                control2: CGPoint(x: cx - ox, y: cy + ry)
            )
            p.addCurve(
                to: CGPoint(x: cx + rx, y: cy),
                control1: CGPoint(x: cx + ox, y: cy + ry),
                control2: CGPoint(x: cx + rx, y: cy + oy)
            )
            p.addCurve(
                to: CGPoint(x: cx, y: cy - ry),
                control1: CGPoint(x: cx + rx, y: cy - oy),
                control2: CGPoint(x: cx + ox, y: cy - ry)
            )
            p.addCurve(
                to: CGPoint(x: cx - rx, y: cy),
                control1: CGPoint(x: cx - ox, y: cy - ry),
                control2: CGPoint(x: cx - rx, y: cy - oy)
            )
        }
        p.closeSubpath()
        return p
    }
    
    func makeStar(cx: CGFloat, cy: CGFloat, rOuter: CGFloat, rInner: CGFloat, points: Int) -> CGPath {
        let p = CGMutablePath()
        let startAngle: CGFloat = -.pi / 2
        let step = (.pi * 2) / CGFloat(points)
        for i in 0..<(points * 2) {
            let r = (i % 2 == 0) ? rOuter : rInner
            let ang = startAngle + CGFloat(i) * (step / 2)
            let x = cx + cos(ang) * r
            let y = cy + sin(ang) * r
            if i == 0 { p.move(to: .init(x: x, y: y)) } else { p.addLine(to: .init(x: x, y: y)) }
        }
        p.closeSubpath()
        return p
    }
    
    func curve1() -> CGPath {
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 520, y: 140))
        path.addQuadCurve(to: CGPoint(x: 760, y: 140), control: CGPoint(x: 640, y: 60))
        path.addQuadCurve(to: CGPoint(x: 520, y: 140), control: CGPoint(x: 640, y: 220))
        return path
    }
    
    func addEllipse(_ cx: CGFloat, _ cy: CGFloat, _ rx: CGFloat, _ ry: CGFloat) -> CGPath {
        let kappa: CGFloat = 0.5522847498307936  // 4*(sqrt(2)-1)/3
        let ox = rx * kappa
        let oy = ry * kappa
        let path = CGMutablePath()
        path.move(to: CGPoint(x: cx - rx, y: cy))
        path.addCurve(
            to: CGPoint(x: cx, y: cy - ry),
            control1: CGPoint(x: cx - rx, y: cy - oy),
            control2: CGPoint(x: cx - ox, y: cy - ry)
        )
        path.addCurve(
            to: CGPoint(x: cx + rx, y: cy),
            control1: CGPoint(x: cx + ox, y: cy - ry),
            control2: CGPoint(x: cx + rx, y: cy - oy)
        )
        path.addCurve(
            to: CGPoint(x: cx, y: cy + ry),
            control1: CGPoint(x: cx + rx, y: cy + oy),
            control2: CGPoint(x: cx + ox, y: cy + ry)
        )
        path.addCurve(
            to: CGPoint(x: cx - rx, y: cy),
            control1: CGPoint(x: cx - ox, y: cy + ry),
            control2: CGPoint(x: cx - rx, y: cy + oy)
        )
        path.closeSubpath()
        return path
    }
}

@main
struct AppMain {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Pathfinder Metal App"
        let content = MetalView(frame: window.contentView!.bounds)
        content.autoresizingMask = [.width, .height]
        window.contentView = content
        window.center()
        window.makeKeyAndOrderFront(nil)
        app.activate(ignoringOtherApps: true)
        app.run()
    }
}
