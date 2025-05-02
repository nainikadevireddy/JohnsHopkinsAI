### Task 3: Requirements Specification

#### Task 3.1: Create a requirements list that includes both functional and non-functional requirements. Include a detailed explanation of the requirements, and research approaches and technologies you may implement to address these requirements.

#### NOTE: A clear paragraph for each of the requirements will do here. You may use this more detailed resource as a guide.

--- 

#### 1. The system should precisely detect fraudulent transactions in real-time.

  The system must analyze transaction data as it occurs and identify potential fraudulent activty in real-time. This requires sophisticated algorithms capable of processing large volumes of data with minimum latency. Real-time detection allows for prompt alerts to the user and the fraud detection team, minimizing financial loss and allow for proactive review. Delayed action can also come to the annoyance of the customers. Machine learning algorithms can be used to detect the fraudulent transactions. A pre-trained, efficient model can produce rapid predictions for real-time transactions. 

#### 2. The system should generate alerts for flagged fraudulent transactions for the fraud detection team and the client.

  When a transaction is flagged as potentially fraudulent, the system must notify both the affected client and the fraud detection team. The immediate alert allows for quick review and action to take place. Knowing whether or not a flagged case is actually fraudulent or not is important to take the appropriate action of cancelling or approving a transaction, as well as important for evaluating the performance metrics of the model. Implementing a dashboard for the fraud detection team to seamlessly access all information pertaining to an alert is important. Similarly, a push-notification on the banking app, text message or email, as preferred by the customer, to the customer will help for easy verification of the fraudulent charge. 


#### 3. The model must identify fraudulent transactions with precision above or at 40%, and recall above or at 85%.

  Achieving a precision of at least 40% and a recall of at least 85% ensures that the system performs equally well or better than the previous fraud detection system at release. With these perfromance metrics, the system correctly identifies a significant portion of fraudulent transactions while minimizing false positives. Precision measures the accuracy of positive predictions, while recall measures the ability to identify all relevant instances. Training and fine-tuning the machine learning algorithm used to detect fradulent cases allows for an optimal model. Regularly validating the model using a diverse dataset can help maintain and improve precision and recall.

 
#### 4. The system should integrate with SecureBank's existing databases of transaction data and client information, and integrate with the existing IT infrastructure.

  Seamless integrations with SecureBank's existing infrastructure ensures that the system has access to the necessary data require to operate. Failure to do so may complicate the integration process, requiring multiple data transformations to generate a connected system. APIs can be used to integrate with existing databases and access user and transaction information, as well as return fraudulent charge information. 


#### 5. The system should facilitate the review process for flagged transactions until the case is resolved.

  The system should not only predict the fraudulent charges, but also facilitate the entire start-to-end process of a fraud investigation. Once flagged, the fraudulent charge must be verified by the fraud detection team and the customer. Once a final resolution is made, the system should be able to cancel or reverse fraudulent transactions or signal for non-fraudulent transaction to be approved. This may also involve blocking or unblocking a customer's bank card, updating system performance metric, reviewing other recent transactions by the customer for potential repeated fraud, etc. 


#### 6. The system must be scalable to the numerous transactions concurred by the hundreds of thousands of SecureBank clients.

  As SecureBank hosts hundreds of thousands of clients, and numerous transactions on a daily basis, scalability of the system is very important. The system should be able to evaluate every transaction processed by the bank seamlessly and efficiently. Cloud-based platforms often allow for scalable infrastructure. 

#### 7. The system should provide a user-friendly interface for customers to confirm flagged transactions.

  A user-friendly interface allows for user satisfaction. User satisfaction, in turn, results in high profitability and incentive to keep existing clients. A user-friendly interface allows customers to easily verify and confirm whether a flagged transaction is fraudulent. This reduces customer frustration and speeds up the resolution process. Mobile and web apps with friendly GUI allows for an intuitive, accessible system for users.   

#### 8. The system should operate effectively, efficiently, and ethically while complying with regulatory requirements and internal policies.

  The system must meet performance standards and adhere to ethical guidelines and regulatory requirements. Compliance ensures legal integrity, that is crucial for a bank. Implementing regular, scheduled audits and compliance checks can help keep the system in check. Ethical AI practices are also important to follow, and performing bias audits on the machine learning models can help maintain ethical practices. 


#### 9. The system should generate metric repots on the system's performance, documenting false positives, false negatives and system performance, precision and recall.

  Regular reporting of the system's performance can allow for monitoring of the system's effectiveness. Previously, the deterioration of the model was not known until there was a downturn in business results, and an increase in client complaints. Regular reports allows the team to update and re-evaluate when the model may deteriorate in performance. Scripts can be used to generate these reports, and visualize the performance metrics of the model. 

#### 10. The system should document the algorithmic process and decision-making process for potential audit review.

  Comprehensive documentation of the algorithmic and decision-making processes ensures transparency and accountability. This is crucial for audits and regulatory reviews, demonstrating that the system operates fairly and accurately. Implement logging mechanisms to record decision-making processes and algorithmic actions, ensuring that all steps are traceable and auditable.



