using Drumwise.Application.Entities;
using Drumwise.Infrastructure.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace Drumwise.Infrastructure.Data;
using Microsoft.AspNetCore.Builder;

public static class IdentityDbInitializerExtensions
{
    public static async Task InitializeIdentityDatabaseAsync(this WebApplication app)
    {
        using var scope = app.Services.CreateScope();
        var initializer = scope.ServiceProvider.GetRequiredService<AppIdentityDbContextInitializer>();
        await initializer.InitializeAsync();
        await initializer.SeedAsync();
    }
}

public class AppIdentityDbContextInitializer(
    AppIdentityDbContext context,
    UserManager<ApplicationUser> userManager,
    RoleManager<IdentityRole> roleManager)
{
    public async Task InitializeAsync()
    {
        try
        {
            await context.Database.MigrateAsync();
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            throw;
        }
    }
    
    public async Task SeedAsync()
    {
        // Default roles
        var roles = new List<IdentityRole>()
        {
            new(Roles.Administrator),
            new(Roles.Student),
            new(Roles.Teacher)
        };
        
        foreach (var role in roles.Where(role => roleManager.Roles.All(r => r.Name != role.Name)))
        {
            await roleManager.CreateAsync(role);
        }

        // Default users
        var administrator = new ApplicationUser { UserName = "administrator@localhost", Email = "administrator@localhost" };

        if (userManager.Users.All(u => u.UserName != administrator.UserName))
        {
            await userManager.CreateAsync(administrator, "Administrator1!");
            if (!string.IsNullOrWhiteSpace(roles[0].Name))
            {
                await userManager.AddToRolesAsync(administrator, new [] { roles[0].Name }!);
            }
        }
    }
}