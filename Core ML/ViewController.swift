//
//  ViewController.swift
//  Core ML
//
//  Created by 灿灿 on 2018/6/12.
//  Copyright © 2018年 HZSS. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    let model = MobileNet()
    let imageView = UIImageView()
    
    @IBOutlet weak var identifyBtn: UIButton!
    
    @IBOutlet weak var resultLabel: UILabel!
    
    @IBOutlet weak var probLabel: UILabel!
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    @IBAction func getPic(_ sender: Any) {
        let photoPicker = UIImagePickerController()
        photoPicker.delegate = self
        photoPicker.allowsEditing = true
        photoPicker.sourceType = .photoLibrary
        
        self.present(photoPicker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        let image = info[UIImagePickerControllerOriginalImage] as! UIImage
//        self.imageView.image = image
        
        let pixelBuffer = image.pixelBuffer(width: 224, height: 224)
        
        let output = try?model.prediction(image: pixelBuffer!)
        
        let probs = output?.classLabelProbs.sorted { $0.value > $1.value }
        
        if let prob = probs?.first {
            resultLabel.text = "结果：\(prob.key) "
            //"\(prob.value * 100) "
            let probText = String(format:"%.2f %", prob.value * 100)
            
            probLabel.text = "相似度：\(probText)%"
        }
        
        self.dismiss(animated: true, completion: nil)
    }
    

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

