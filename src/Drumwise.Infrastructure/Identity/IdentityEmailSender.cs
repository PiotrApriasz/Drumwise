using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Settings;
using MailKit.Net.Smtp;
using Microsoft.AspNetCore.Identity;
using MimeKit;

namespace Drumwise.Infrastructure.Identity;

public class IdentityEmailSender(SmtpSettings smtpSettings, IFileService fileService) : IEmailSender<ApplicationUser>
{
    public async Task SendConfirmationLinkAsync(ApplicationUser user, string email, string confirmationLink)
    {
        var message = new MimeMessage();
        message.From.Add(new MailboxAddress("Drumwise", smtpSettings.User));
        message.To.Add(new MailboxAddress(user.UserName, email));
        message.Subject = "Confirm your email for Drumwise";
        
        var emailConfirmationTemplate = await fileService
            .GetTemplateAsync("EmailTemplates", "EmailConfirmationTemplate", "html").ConfigureAwait(false);

        emailConfirmationTemplate = emailConfirmationTemplate
            .Replace("[EMAIL_ADDRESS]", email)
            .Replace("[CONFIRMATION_LINK]", confirmationLink);

        message.Body = new TextPart("html")
        {
            Text = emailConfirmationTemplate
        };

        using var client = new SmtpClient();
        await client.ConnectAsync(smtpSettings.Server, smtpSettings.Port, true).ConfigureAwait(false);
        await client.AuthenticateAsync(smtpSettings.User, smtpSettings.Password).ConfigureAwait(false);
        await client.SendAsync(message).ConfigureAwait(false);
        await client.DisconnectAsync(true).ConfigureAwait(false);
    }

    public Task SendPasswordResetLinkAsync(ApplicationUser user, string email, string resetLink)
    {
        throw new NotImplementedException();
    }

    public Task SendPasswordResetCodeAsync(ApplicationUser user, string email, string resetCode)
    {
        throw new NotImplementedException();
    }
}