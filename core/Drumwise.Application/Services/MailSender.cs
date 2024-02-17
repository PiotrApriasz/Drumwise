using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Settings;
using MimeKit;
using MailKit.Net.Smtp;

namespace Drumwise.Application.Services;

public class MailSender(SmtpSettings smtpSettings) : IMailSender
{
    public async Task SendMailAsync(string email, string subject, string body, string bodyType)
    {
        var message = new MimeMessage();
        message.From.Add(new MailboxAddress("Drumwise", smtpSettings.User));
        message.To.Add(new MailboxAddress(email, email));
        message.Subject = subject;
        
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
}