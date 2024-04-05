using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Settings;
using MailKit.Net.Smtp;
using MimeKit;

namespace Drumwise.Application.Services.Mailing;

public class MailSender(SmtpSettings smtpSettings, IFileService fileService) : IMailSender
{
    public async Task SendMailAsync(string email, string subject, IMailData mailData, string bodyType = "html")
    {
        var message = new MimeMessage();
        message.From.Add(new MailboxAddress("Drumwise", smtpSettings.User));
        message.To.Add(new MailboxAddress(email, email));
        message.Subject = subject;

        var template = await GetTemplateForEmailData(mailData).ConfigureAwait(false);
        var body = FillEmailTemplateWithData(template, mailData);
        
        message.Body = new TextPart(bodyType)
        {
            Text = body
        };
        
        using var client = new SmtpClient();
        await client.ConnectAsync(smtpSettings.Server, smtpSettings.Port, true).ConfigureAwait(false);
        await client.AuthenticateAsync(smtpSettings.User, smtpSettings.Password).ConfigureAwait(false);
        await client.SendAsync(message).ConfigureAwait(false);
        await client.DisconnectAsync(true).ConfigureAwait(false);
    }
    
    private async Task<string> GetTemplateForEmailData(IMailData data)
    {
        var mailDataName = data.GetType().Name;
        var dataIndex = mailDataName.IndexOf("Data", StringComparison.Ordinal);
        var templateName = $"{mailDataName[..dataIndex]}Template.html";

        return await fileService.GetTemplateAsync("EmailTemplates", templateName);
    }
    
    private static string FillEmailTemplateWithData(string template, IMailData data)
    {
        var dataType = data.GetType();
        var properties = dataType.GetProperties();

        foreach (var property in properties)
        {
            var placeholder = $"[{property.Name}]";
            var value = property.GetValue(data)?.ToString();
            template = template.Replace(placeholder, value ?? string.Empty);
        }

        return template;
    }
}