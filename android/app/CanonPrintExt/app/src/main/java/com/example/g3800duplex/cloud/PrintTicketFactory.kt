package com.example.g3800duplex.cloud

import com.example.g3800duplex.print.PrintPaperSettings

/**
 * PrintTicket XML for Canon CNPS cloud convert (aligned with official raw printticket).
 */
object PrintTicketFactory {
    @Deprecated("Use jpeg300(deviceName, paper)", ReplaceWith("jpeg300(deviceName)"))
    fun a4Jpeg300(deviceName: String = "G3800"): String =
        jpeg300(deviceName, PrintPaperSettings())

    fun jpeg300(
        deviceName: String = "G3800",
        paper: PrintPaperSettings = PrintPaperSettings(),
    ): String {
        val safe = deviceName.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        val mediaSize = paper.size.cloudMediaSize
        val mediaClass = paper.media.cloudMediaClass
        return """
            <?xml version="1.0" encoding="UTF-8"?>
            <cpf2:JobTicket version="210000"
            		xmlns:cpf2="http://www.canon.com/ns/dp/cpc/v200/common/cptfw"
            		xmlns:cpk2="http://www.canon.com/ns/dp/cpc/v200/common/cptkw">
            	<cpf2:Job>
            		<cpf2:Property name="cpk2:OutputDeviceName">
            			<cpf2:Value>$safe</cpf2:Value>
            		</cpf2:Property>
            		<cpf2:PrintTicket version="210000">
            			<cpf2:SelectiveParam name="cpk2:MediaTypeClass">
            				<cpf2:Option>$mediaClass</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:ValueParam name="cpk2:ResolutionX">
            				<cpf2:Value>300</cpf2:Value>
            			</cpf2:ValueParam>
            			<cpf2:ValueParam name="cpk2:ResolutionY">
            				<cpf2:Value>300</cpf2:Value>
            			</cpf2:ValueParam>
            			<cpf2:SelectiveParam name="cpk2:OutputColor">
            				<cpf2:Option>cpk2:Color</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:SelectiveParam name="cpk2:OutputMediaSize">
            				<cpf2:Option>$mediaSize</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:SelectiveParam name="cpk2:Orientation">
            				<cpf2:Option>cpk2:Portrait</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:SelectiveParam name="cpk2:LayoutType">
            				<cpf2:Option>cpk2:Normal</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:SelectiveParam name="cpk2:PageScalingType">
            				<cpf2:Option>cpk2:FitPage</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:ValueParam name="cpk2:DocumentNUP">
            				<cpf2:Value>1</cpf2:Value>
            			</cpf2:ValueParam>
            			<cpf2:SelectiveParam name="cpk2:Duplex">
            				<cpf2:Option>cpk2:OneSided</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:SelectiveParam name="cpk2:PageOrder">
            				<cpf2:Option>cpk2:Normal</cpf2:Option>
            			</cpf2:SelectiveParam>
            			<cpf2:ValueParam name="cpk2:Copies">
            				<cpf2:Value>1</cpf2:Value>
            			</cpf2:ValueParam>
            			<cpf2:SelectiveParam name="cpk2:RenderingQuality">
            				<cpf2:Option>cpk2:Middle</cpf2:Option>
            			</cpf2:SelectiveParam>
            		</cpf2:PrintTicket>
            	</cpf2:Job>
            </cpf2:JobTicket>
        """.trimIndent()
    }
}
